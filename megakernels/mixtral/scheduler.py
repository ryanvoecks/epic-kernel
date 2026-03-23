import math

import torch

from megakernels.instructions import NoOp
from megakernels.mixtral.instructions import (
    ExpertDownProjAccum,
    ExpertUpGateSiLU,
    Mixtral_AttnReduction,
    Mixtral_OProj,
    Mixtral_PartialAttn,
    Mixtral_QKV,
    Mixtral_RMS_LM_Head,
    MixtralGlobals,
    MoE_Router,
)
from megakernels.mixtral.model import MixtralForCausalLM
from megakernels.scheduler import DAG_Node, ScheduleBuilder
from megakernels.utils import assert_div, get_sm_count

B200_SM_COUNT = 160


def pick_num_attention_partitions(
    prompt_len: int, ntok: int, num_kv_heads: int, device
) -> int:
    min_chunk_size = 256
    full_len = prompt_len + ntok
    num_divisions = math.ceil(full_len / min_chunk_size)
    sm_count = min(get_sm_count(device), B200_SM_COUNT)
    num_partitions = min(num_divisions, sm_count // num_kv_heads)
    return max(1, num_partitions)


def make_globals(
    model: MixtralForCausalLM,
    skip_attn_reduction: bool | None = None,
    seq_len: int = 0,
) -> MixtralGlobals:
    config = model.config
    device = model.device
    dtype = model.dtype
    sp = model.stacked_params

    if skip_attn_reduction is None:
        actual_seq_len = seq_len + 1
        num_partitions = pick_num_attention_partitions(
            actual_seq_len, 0, config.num_key_value_heads, device
        )
        skip_attn_reduction = num_partitions == 1

    def buf(shape, buf_dtype=dtype):
        return torch.zeros(shape, device=device, dtype=buf_dtype)

    max_attn_partitions = get_sm_count(device)

    num_experts = config.num_local_experts
    num_kv_heads = config.num_key_value_heads
    num_attn_heads = config.num_attention_heads

    # Barriers: [num_layers, num_opcodes=10, max_heads_or_experts]
    # Third dim = max(num_attn_heads + 2*num_kv_heads, num_experts)
    barrier_head_dim = max(num_attn_heads + 2 * num_kv_heads, num_experts)
    barriers = torch.zeros(
        [config.num_hidden_layers, 10, barrier_head_dim],
        dtype=torch.int32,
        device=device,
    )

    head_dim = config.hidden_size // config.num_attention_heads
    matvec_red = math.gcd(config.intermediate_size, config.hidden_size)

    # Dummy tensors for unused BaseGlobals MLP fields
    _dummy = torch.empty(0, device=device, dtype=dtype)

    return MixtralGlobals(
        # BaseGlobals attention weights
        qkv_proj_weights=sp.qkv_proj,
        attn_ln_weights=sp.attn_ln_weight,
        o_proj_weights=sp.o_proj,
        # BaseGlobals MLP weights (unused for Mixtral — pass dummies)
        mlp_ln_weights=_dummy,
        up_proj_weights=_dummy,
        gate_proj_weights=_dummy,
        down_proj_weights=_dummy,
        # BaseGlobals LM head
        lm_head_norm_weights=model.lm_head.input_norm.weight,
        lm_head_weights=model.lm_head.lm_head.weight,
        # KV cache
        k_cache=model.stacked_kv_cache[0],
        v_cache=model.stacked_kv_cache[1],
        # RoPE
        rope_cos=model.model.rope_cos,
        rope_sin=model.model.rope_sin,
        # Model constants
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        head_dim=head_dim,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        vocab_size=config.vocab_size,
        attn_scale=1.0 / math.sqrt(head_dim),
        rms_norm_eps=config.rms_norm_eps,
        device=device,
        barriers=barriers,
        pos_id=seq_len,
        # Attention activation buffers
        hidden_states=buf(config.hidden_size),
        post_ln_rope_q=buf(config.hidden_size),
        attn_out=buf(config.hidden_size),
        attn_out_intermediates=buf(
            [config.num_attention_heads, max_attn_partitions, head_dim],
            buf_dtype=torch.float32,
        ),
        attn_lse_intermediates=buf(
            [config.num_attention_heads, max_attn_partitions],
            buf_dtype=torch.float32,
        ),
        # MoE activation buffers
        expert_silu_out=buf([num_experts, config.intermediate_size]),
        router_normed_hidden=buf(config.hidden_size),
        logits=buf(config.vocab_size),
        selected_expert_indices=torch.zeros(
            config.num_experts_per_tok, dtype=torch.int32, device=device
        ),
        selected_expert_scores=torch.zeros(
            config.num_experts_per_tok, device=device, dtype=dtype
        ),
        # MoE stacked weights
        router_weights=sp.router_weight,
        expert_gate_weights=sp.expert_gate,
        expert_up_weights=sp.expert_up,
        expert_down_weights=sp.expert_down,
        ffn_ln_weights=sp.ffn_ln_weight,
        # MoE constants
        num_experts=num_experts,
        num_experts_per_tok=config.num_experts_per_tok,
        # Flags & block sizes
        skip_attn_reduction=skip_attn_reduction,
        expert_proj_block_size=16,
        down_proj_block_size=16,
        o_proj_block_size=16,
        lm_head_block_size=16,
        qkv_block_size=16,
        attn_kv_block_size=16,
        matvec_reduction_size=matvec_red,
        attn_reduction_size=config.num_attention_heads // config.num_key_value_heads,
    )


# ---------------------------------------------------------------------------
# Per-stage scheduling helpers
# ---------------------------------------------------------------------------

def schedule_qkv(globs: MixtralGlobals, layer_idx: int) -> list[Mixtral_QKV]:
    qkv_outdim = (globs.num_attention_heads + 2 * globs.num_kv_heads) * globs.head_dim
    num_qkv_blocks = assert_div(qkv_outdim, globs.qkv_block_size)
    # Cap effective SMs to block count so every instruction covers ≥ 1 block.
    effective_sms = min(globs.sm_count(), num_qkv_blocks)
    blocks_per_sm = num_qkv_blocks / effective_sms

    instructions = []
    for sm_idx in range(effective_sms):
        start = round(sm_idx * blocks_per_sm)
        end = round((sm_idx + 1) * blocks_per_sm)
        instructions.append(
            Mixtral_QKV(layer_idx=layer_idx, start_output_block_idx=start, end_output_block_idx=end)
        )
    return instructions


def schedule_expert_upgate(globs: MixtralGlobals, layer_idx: int) -> list[ExpertUpGateSiLU]:
    """Schedule UpGate for all 8 experts (inactive ones are NoOps in CUDA)."""
    num_up_blocks = assert_div(globs.intermediate_size, globs.expert_proj_block_size)
    sm_count = globs.sm_count()
    num_experts = globs.num_experts

    # Distribute blocks across SMs, cycling through experts.
    # Cap effective SMs to block count so every instruction covers ≥ 1 block.
    effective_sms = min(sm_count, num_up_blocks)
    blocks_per_sm = num_up_blocks / effective_sms

    instructions = []
    for expert_idx in range(num_experts):
        for sm_idx in range(effective_sms):
            start = round(sm_idx * blocks_per_sm)
            end = round((sm_idx + 1) * blocks_per_sm)
            if start >= end:
                continue
            instructions.append(
                ExpertUpGateSiLU(
                    layer_idx=layer_idx,
                    expert_idx=expert_idx,
                    start_block_idx=start,
                    num_blocks=end - start,
                )
            )
    return instructions


def schedule_expert_downproj(
    globs: MixtralGlobals, layer_idx: int
) -> list[ExpertDownProjAccum]:
    """Schedule DownProj for all 8 experts."""
    num_down_blocks = assert_div(globs.hidden_size, globs.down_proj_block_size)
    num_col_splits = assert_div(globs.intermediate_size, globs.matvec_reduction_size)
    sm_count = globs.sm_count()
    num_experts = globs.num_experts

    jobs = []
    for expert_idx in range(num_experts):
        for col_idx in range(num_col_splits):
            for down_block_idx in range(num_down_blocks):
                jobs.append((expert_idx, col_idx, down_block_idx))

    # Cap effective SMs to the number of jobs so every SM gets ≥ 1 job.
    effective_sms = min(sm_count, len(jobs))
    instructions = []
    num_assigned = 0
    for sm_idx in range(effective_sms):
        jobs_left = len(jobs) - num_assigned
        sms_left = effective_sms - sm_idx
        jobs_per_sm = jobs_left / sms_left

        jobs_for_sm = round(jobs_per_sm)
        raw = jobs[num_assigned : num_assigned + jobs_for_sm]

        # Ensure all jobs for this SM share the same (expert_idx, col_idx)
        expert_idx, col_idx = raw[0][0], raw[0][1]
        sliced = [j for j in raw if j[0] == expert_idx and j[1] == col_idx]
        assert len(sliced) > 0

        start_block = sliced[0][2]
        block_indices = [j[2] for j in sliced]
        assert block_indices == list(range(start_block, start_block + len(sliced)))

        instructions.append(
            ExpertDownProjAccum(
                layer_idx=layer_idx,
                expert_idx=expert_idx,
                start_block_idx=start_block,
                end_block_idx=start_block + len(sliced),
                reduction_block_idx=col_idx,
            )
        )
        num_assigned += len(sliced)

    return instructions


def schedule_lm_head(globs: MixtralGlobals) -> list[Mixtral_RMS_LM_Head]:
    num_logit_blocks = assert_div(globs.vocab_size, globs.lm_head_block_size)
    sm_count = globs.sm_count()
    effective_sms = min(sm_count, num_logit_blocks)
    blocks_per_sm = num_logit_blocks / effective_sms

    instructions = []
    for sm_idx in range(effective_sms):
        start = round(sm_idx * blocks_per_sm)
        end = round((sm_idx + 1) * blocks_per_sm)
        instructions.append(
            Mixtral_RMS_LM_Head(start_output_block_idx=start, end_output_block_idx=end)
        )
    return instructions


# ---------------------------------------------------------------------------
# DAG construction
# ---------------------------------------------------------------------------

def make_dag(
    globs: MixtralGlobals,
    stop_after_op: str | None = None,
    layer_limit: int | None = None,
):
    nodes: list[DAG_Node] = []
    nlayers = layer_limit if layer_limit is not None else globs.num_hidden_layers

    last_outputs: list[DAG_Node] = []
    for layer_idx in range(nlayers):
        new_nodes, last_outputs = make_dag_layer(
            globs=globs,
            layer_idx=layer_idx,
            prev_layer_outputs=last_outputs,
            stop_after_op=stop_after_op,
        )
        nodes.extend(new_nodes)

    if nlayers == globs.num_hidden_layers and stop_after_op is None:
        lm_head_nodes = [
            DAG_Node(ins, last_outputs) for ins in schedule_lm_head(globs)
        ]
        nodes.extend(lm_head_nodes)
        last_outputs = lm_head_nodes

    end_node = DAG_Node(NoOp(), last_outputs)
    return nodes, end_node


def make_dag_layer(
    globs: MixtralGlobals,
    layer_idx: int,
    prev_layer_outputs: list[DAG_Node],
    stop_after_op: str | None = None,
):
    actual_seq_len = globs.pos_id + 1
    num_partitions = pick_num_attention_partitions(
        actual_seq_len, 0, globs.num_kv_heads, globs.device
    )
    globs.skip_attn_reduction = num_partitions == 1

    new_nodes: list[DAG_Node] = []

    # --- QKV ---
    qkv_instructions = schedule_qkv(globs, layer_idx)
    qkv_nodes = [DAG_Node(ins, prev_layer_outputs) for ins in qkv_instructions]

    # Map block_idx → DAG node for dependency resolution
    qkv_deps: dict[tuple, DAG_Node] = {}
    for node in qkv_nodes:
        ins = node.instruction
        for block_idx in ins.block_indices():
            qkv_deps[(layer_idx, ins.opcode(), block_idx)] = node

    new_nodes.extend(qkv_nodes)
    if stop_after_op == "qkv":
        return new_nodes, qkv_nodes

    # --- Partial Attention ---
    partial_nodes: list[DAG_Node] = []
    bph = globs.head_dim // globs.qkv_block_size  # blocks per head

    for kv_head_idx in range(globs.num_kv_heads):
        for partial_idx in range(num_partitions):
            ins = Mixtral_PartialAttn(
                layer_idx=layer_idx,
                kv_head_idx=kv_head_idx,
                num_partials=num_partitions,
                partial_idx=partial_idx,
            )
            gqa_ratio = globs.num_attention_heads // globs.num_kv_heads
            k_start_dim = (globs.num_attention_heads + kv_head_idx) * globs.head_dim
            v_start_dim = (globs.num_attention_heads + globs.num_kv_heads + kv_head_idx) * globs.head_dim

            block_indices = (
                list(range(k_start_dim // globs.qkv_block_size,
                           k_start_dim // globs.qkv_block_size + bph))
                + list(range(v_start_dim // globs.qkv_block_size,
                             v_start_dim // globs.qkv_block_size + bph))
            )
            dep_set = {
                qkv_deps[(layer_idx, Mixtral_PartialAttn.prev_opcode(), bi)]
                for bi in block_indices
            }
            partial_nodes.append(DAG_Node(ins, list(dep_set)))

    new_nodes.extend(partial_nodes)
    if stop_after_op == "partial":
        return new_nodes, partial_nodes

    # --- Attention Reduction (skipped when num_partitions == 1) ---
    gqa_ratio = globs.num_attention_heads // globs.num_kv_heads
    if num_partitions > 1:
        reduction_nodes: list[DAG_Node] = []
        for kv_head_idx in range(globs.num_kv_heads):
            ins = Mixtral_AttnReduction(
                layer_idx=layer_idx,
                head_start_idx=kv_head_idx * gqa_ratio,
                num_partials=num_partitions,
                is_terminal=True,
                reduction_list=list(range(num_partitions)),
                output_partial_idx=None,
            )
            deps = [n for n in partial_nodes if n.instruction.kv_head_idx == kv_head_idx]
            reduction_nodes.append(DAG_Node(ins, deps))
        new_nodes.extend(reduction_nodes)
        o_proj_deps = reduction_nodes
    else:
        o_proj_deps = partial_nodes

    if stop_after_op == "reduction":
        return new_nodes, o_proj_deps

    # --- Output Projection ---
    num_o_blocks = assert_div(globs.hidden_size, globs.o_proj_block_size)
    o_proj_nodes = [
        DAG_Node(
            Mixtral_OProj(
                layer_idx=layer_idx,
                start_block_idx=b,
                end_block_idx=b + 1,
                reduction_block_idx=0,
            ),
            o_proj_deps,
        )
        for b in range(num_o_blocks)
    ]
    new_nodes.extend(o_proj_nodes)
    if stop_after_op == "oproj":
        return new_nodes, o_proj_nodes

    # --- MoE Router (single node — sequential bottleneck) ---
    router_node = DAG_Node(MoE_Router(layer_idx=layer_idx), o_proj_nodes)
    new_nodes.append(router_node)
    if stop_after_op == "router":
        return new_nodes, [router_node]

    # --- Expert UpGate (all 8 experts; inactive ones skip in CUDA) ---
    upgate_instructions = schedule_expert_upgate(globs, layer_idx)
    # Group by expert for dependency on DownProj
    upgate_nodes_by_expert: dict[int, list[DAG_Node]] = {e: [] for e in range(globs.num_experts)}
    upgate_nodes: list[DAG_Node] = []
    for ins in upgate_instructions:
        node = DAG_Node(ins, [router_node])
        upgate_nodes.append(node)
        upgate_nodes_by_expert[ins.expert_idx].append(node)
    new_nodes.extend(upgate_nodes)
    if stop_after_op == "expert_upgate":
        return new_nodes, upgate_nodes

    # --- Expert DownProj (each depends on same-expert UpGate nodes) ---
    # To avoid races on hidden_states, serialize DownProj: expert 0 before expert 1.
    # Implement by making expert i+1 depend on all of expert i's DownProj nodes.
    downproj_instructions = schedule_expert_downproj(globs, layer_idx)

    # Group downproj instructions by expert
    downproj_by_expert: dict[int, list[ExpertDownProjAccum]] = {
        e: [] for e in range(globs.num_experts)
    }
    for ins in downproj_instructions:
        downproj_by_expert[ins.expert_idx].append(ins)

    downproj_nodes: list[DAG_Node] = []
    prev_expert_downproj_nodes: list[DAG_Node] = []

    for expert_idx in range(globs.num_experts):
        expert_upgate_deps = upgate_nodes_by_expert[expert_idx]
        # Also depend on previous expert's DownProj to serialize writes to hidden_states
        deps = expert_upgate_deps + prev_expert_downproj_nodes

        expert_down_nodes: list[DAG_Node] = []
        for ins in downproj_by_expert[expert_idx]:
            node = DAG_Node(ins, deps)
            expert_down_nodes.append(node)
            downproj_nodes.append(node)

        new_nodes.extend(expert_down_nodes)
        prev_expert_downproj_nodes = expert_down_nodes

    if stop_after_op in ("downproj", "expert_downproj"):
        return new_nodes, downproj_nodes

    assert stop_after_op is None
    return new_nodes, downproj_nodes


# ---------------------------------------------------------------------------
# ScheduleBuilder
# ---------------------------------------------------------------------------

class MixtralLatencyScheduleBuilder(ScheduleBuilder):
    @classmethod
    def make_globals(cls, model: MixtralForCausalLM, seq_len: int = 0) -> MixtralGlobals:
        return make_globals(model, seq_len=seq_len)

    @classmethod
    def make_dag(cls, globs, stop_after_op=None, layer_limit=None):
        return make_dag(globs, stop_after_op, layer_limit)
