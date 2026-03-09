import math

import torch

from megakernels.demos.latency.instructions import (
    AttentionReduction,
    DownProjResidual,
    Globals,
    Instruction,
    LayerNorm_QKV_MatVecRopeAppend,
    LayerNormDoubleMatVecSiLU,
    O_ProjResidual,
    PartialAttention,
    RMS_LM_Head,
)
from megakernels.instructions import NoOp
from megakernels.llama import LlamaForCausalLM
from megakernels.scheduler import DAG_Node, ScheduleBuilder
from megakernels.utils import assert_div, get_sm_count


B200_SM_COUNT = 160


def pick_num_attention_partitions(prompt_len: int, ntok: int, num_kv_heads: int, device: torch.device):
    min_chunk_size = 256
    full_len = prompt_len + ntok

    num_divisions = math.ceil(full_len / min_chunk_size)

    sm_count = min(get_sm_count(device), B200_SM_COUNT)
    num_attention_partitions = min(num_divisions, sm_count // num_kv_heads)

    assert num_attention_partitions >= 1

    return num_attention_partitions


def make_globals(
    model: LlamaForCausalLM,
    skip_attn_reduction: bool | None = None,
    seq_len: int = 0,
):
    config = model.config
    device = model.device
    dtype = model.dtype

    if skip_attn_reduction is None:
        # Compute from seq_len so that with_new_globals (which doesn't
        # re-run make_dag) still gets the correct value.
        actual_seq_len = seq_len + 1  # mirrors make_dag_layer: pos_id + 1
        num_partitions = pick_num_attention_partitions(
            actual_seq_len, 0, config.num_key_value_heads, device
        )
        skip_attn_reduction = num_partitions == 1

    def make_buffer(shape, buffer_dtype=dtype):
        return torch.zeros(shape, device=device, dtype=buffer_dtype)

    stacked_params = model.stacked_params

    max_attn_partitions = get_sm_count(device)

    barriers = torch.zeros(
        [
            config.num_hidden_layers,
            10,  # more than the number of opcodes we have
            config.num_attention_heads + config.num_key_value_heads * 2,
        ],
        dtype=torch.int32,
        device=device,
    )

    return Globals(
        # model params
        qkv_proj_weights=stacked_params.qkv_proj,
        o_proj_weights=stacked_params.o_proj,
        attn_ln_weights=stacked_params.attn_ln_weight,
        mlp_ln_weights=stacked_params.mlp_ln_weight,
        up_proj_weights=stacked_params.up_proj,
        gate_proj_weights=stacked_params.gate_proj,
        down_proj_weights=stacked_params.down_proj,
        lm_head_norm_weights=model.lm_head.input_norm.weight,
        lm_head_weights=model.lm_head.lm_head.weight,
        k_cache=model.stacked_kv_cache[0],
        v_cache=model.stacked_kv_cache[1],
        rope_cos=model.model.rope_cos,
        rope_sin=model.model.rope_sin,
        # activation buffers
        hidden_states=make_buffer(config.hidden_size),
        post_ln_rope_q=make_buffer(config.hidden_size),
        attn_out=make_buffer(config.hidden_size),
        attn_out_intermediates=make_buffer(
            [config.num_attention_heads, max_attn_partitions, config.head_dim],
            buffer_dtype=torch.float32,
        ),
        attn_lse_intermediates=make_buffer(
            [config.num_attention_heads, max_attn_partitions],
            buffer_dtype=torch.float32,
        ),
        silu_out=make_buffer(config.intermediate_size),
        logits=make_buffer(config.vocab_size),
        # scalars
        pos_id=seq_len,
        attn_scale=1 / math.sqrt(config.head_dim),
        rms_norm_eps=config.rms_norm_eps,
        skip_attn_reduction=skip_attn_reduction,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        # block sizes (computed from config so they work for any model size)
        up_gate_proj_block_size=16,
        down_proj_block_size=16,
        qkv_block_size=16,
        o_proj_block_size=16,
        lm_head_block_size=16,
        # gcd ensures divisibility for both 1B (gcd(8192,2048)=2048) and
        # 3B (gcd(8192,3072)=1024) without hardcoding per-model values.
        matvec_reduction_size=math.gcd(config.intermediate_size, config.hidden_size),
        attn_kv_block_size=16,
        # GQA ratio: num_attention_heads / num_kv_heads
        attn_reduction_size=config.num_attention_heads // config.num_key_value_heads,
        vocab_size=config.vocab_size,
        device=device,
        barriers=barriers,
    )


def schedule_qkv(
    globs: Globals, layer_idx: int
) -> list[LayerNorm_QKV_MatVecRopeAppend]:
    instructions = []

    qkv_outdim = (globs.num_attention_heads + 2 * globs.num_kv_heads) * globs.head_dim

    num_qkv_blocks = assert_div(qkv_outdim, globs.qkv_block_size)
    sm_count = globs.sm_count()

    blocks_per_sm = num_qkv_blocks / sm_count
    assert blocks_per_sm > 1

    for sm_idx in range(sm_count):
        start = round(sm_idx * blocks_per_sm)
        end = round((sm_idx + 1) * blocks_per_sm)
        instructions.append(
            LayerNorm_QKV_MatVecRopeAppend(
                layer_idx=layer_idx,
                start_output_block_idx=start,
                end_output_block_idx=end,
            )
        )

    return instructions


def schedule_upgate(globs: Globals, layer_idx: int):
    instructions: list[Instruction] = []
    num_up_gate_blocks = assert_div(
        globs.intermediate_size, globs.up_gate_proj_block_size
    )

    sm_count = globs.sm_count()

    blocks_per_sm = num_up_gate_blocks / sm_count
    assert blocks_per_sm > 1

    for sm_idx in range(sm_count):
        instructions.append(
            LayerNormDoubleMatVecSiLU(
                layer_idx=layer_idx,
                block_idxs=list(range(sm_idx, num_up_gate_blocks, sm_count)),
            )
        )

    return instructions


def schedule_downproj(globs: Globals, layer_idx: int):
    instructions: list[Instruction] = []

    num_down_blocks = assert_div(globs.hidden_size, globs.down_proj_block_size)
    num_col_splits = assert_div(globs.intermediate_size, globs.matvec_reduction_size)
    sm_count = globs.sm_count()

    jobs = []
    for col_idx in range(num_col_splits):
        for down_block_idx in range(num_down_blocks):
            jobs.append((col_idx, down_block_idx))

    num_assigned_jobs = 0
    for sm_idx in range(sm_count):
        jobs_left = len(jobs) - num_assigned_jobs
        sms_left = sm_count - sm_idx
        jobs_per_sm = jobs_left / sms_left
        assert jobs_per_sm > 1

        jobs_for_this_sm = round(jobs_per_sm)
        raw_sliced_jobs = jobs[num_assigned_jobs : num_assigned_jobs + jobs_for_this_sm]

        col_idx = raw_sliced_jobs[0][0]
        sliced_jobs = [job for job in raw_sliced_jobs if job[0] == col_idx]
        assert len(sliced_jobs) > 0

        start_output_block_idx = sliced_jobs[0][1]
        output_block_indices = [job[1] for job in sliced_jobs]
        assert output_block_indices == list(
            range(
                start_output_block_idx,
                start_output_block_idx + len(sliced_jobs),
            )
        )

        instructions.append(
            DownProjResidual(
                layer_idx=layer_idx,
                start_block_idx=start_output_block_idx,
                end_block_idx=start_output_block_idx + len(sliced_jobs),
                reduction_block_idx=col_idx,
            )
        )

        num_assigned_jobs += len(sliced_jobs)

    return instructions


def schedule_lm_head(globs: Globals):
    instructions: list[Instruction] = []

    num_logit_blocks = assert_div(globs.vocab_size, globs.lm_head_block_size)
    sm_count = globs.sm_count()

    blocks_per_sm = num_logit_blocks / sm_count
    assert blocks_per_sm > 1

    for sm_idx in range(sm_count):
        start = round(sm_idx * blocks_per_sm)
        end = round((sm_idx + 1) * blocks_per_sm)
        instructions.append(
            RMS_LM_Head(start_output_block_idx=start, end_output_block_idx=end)
        )

    return instructions


def make_dag(
    globs: Globals, stop_after_op: str | None = None, layer_limit: int | None = None
):
    nodes: list[DAG_Node] = []

    if layer_limit is not None:
        nlayers = layer_limit
    else:
        nlayers = globs.num_hidden_layers

    last_outputs = []
    for layer_idx in range(nlayers):
        new_nodes, new_outputs = make_dag_layer(
            globs=globs,
            layer_idx=layer_idx,
            prev_layer_outputs=last_outputs,
            stop_after_op=stop_after_op,
        )
        nodes.extend(new_nodes)
        last_outputs = new_outputs

    if nlayers == globs.num_hidden_layers:
        lm_head_instructions = schedule_lm_head(globs)
        lm_head_nodes: list[DAG_Node] = []
        for ins in lm_head_instructions:
            lm_head_nodes.append(DAG_Node(ins, last_outputs))

        nodes.extend(lm_head_nodes)
        last_outputs = lm_head_nodes

    end_node = DAG_Node(NoOp(), last_outputs)

    return nodes, end_node


def make_dag_layer(
    globs: Globals,
    layer_idx: int,
    prev_layer_outputs: list[DAG_Node],
    stop_after_op: str | None = None,
):
    actual_seq_len = globs.pos_id + 1
    num_attention_partitions = pick_num_attention_partitions(actual_seq_len, 0, globs.num_kv_heads, globs.device)
    # num_attention_partitions = 2 # temp hardcode
    globs.skip_attn_reduction = num_attention_partitions == 1

    new_nodes: list[DAG_Node] = []

    # qkv
    qkv_instructions = schedule_qkv(globs, layer_idx)
    qkv_nodes: list[DAG_Node] = []
    for ins in qkv_instructions:
        qkv_nodes.append(DAG_Node(ins, prev_layer_outputs))

    qkv_deps = {}

    for node in qkv_nodes:
        ins: LayerNorm_QKV_MatVecRopeAppend = node.instruction
        for block_idx in ins.block_indices():
            qkv_deps[(layer_idx, ins.opcode(), block_idx)] = node

    new_nodes.extend(qkv_nodes)

    if stop_after_op == "qkv":
        return new_nodes, qkv_nodes

    # partial
    partial_nodes: list[DAG_Node] = []

    for kv_head_idx in range(globs.num_kv_heads):
        for partial_idx in range(num_attention_partitions):
            ins = PartialAttention(
                layer_idx=layer_idx,
                kv_head_idx=kv_head_idx,
                num_partials=num_attention_partitions,
                partial_idx=partial_idx,
            )

            block_indices = []

            k_start_dim = (globs.num_attention_heads + kv_head_idx) * globs.head_dim
            v_start_dim = (
                globs.num_attention_heads + globs.num_kv_heads + kv_head_idx
            ) * globs.head_dim

            dims_per_block = assert_div(globs.head_dim, globs.qkv_block_size)

            k_start_block = k_start_dim // globs.qkv_block_size
            v_start_block = v_start_dim // globs.qkv_block_size

            block_indices.extend(
                list(range(k_start_block, k_start_block + dims_per_block))
            )
            block_indices.extend(
                list(range(v_start_block, v_start_block + dims_per_block))
            )

            dep_set = {
                qkv_deps[(layer_idx, PartialAttention.prev_opcode(), block_idx)]
                for block_idx in block_indices
            }
            deps = list(dep_set)

            partial_nodes.append(DAG_Node(ins, deps))

    new_nodes.extend(partial_nodes)

    if stop_after_op == "partial":
        return new_nodes, partial_nodes

    # reduction (skipped when num_attention_partitions == 1)
    gqa_ratio = globs.num_attention_heads // globs.num_kv_heads
    if num_attention_partitions > 1:
        reduction_nodes: list[DAG_Node] = []
        for kv_head_idx in range(globs.num_kv_heads):
            ins = AttentionReduction(
                layer_idx=layer_idx,
                head_start_idx=kv_head_idx * gqa_ratio,
                num_partials=num_attention_partitions,
                is_terminal=True,
                reduction_list=list(range(num_attention_partitions)),
                output_partial_idx=None,
            )
            deps = [n for n in partial_nodes if n.instruction.kv_head_idx == kv_head_idx]
            reduction_nodes.append(DAG_Node(ins, deps))
        new_nodes.extend(reduction_nodes)
        o_proj_deps = reduction_nodes
    else:
        o_proj_deps = partial_nodes

    # oproj
    num_o_blocks = assert_div(globs.hidden_size, globs.o_proj_block_size)
    o_proj_nodes: list[DAG_Node] = []
    for o_block_idx in range(num_o_blocks):
        ins = O_ProjResidual(
            layer_idx=layer_idx,
            start_block_idx=o_block_idx,
            end_block_idx=o_block_idx + 1,
            reduction_block_idx=0,
        )

        o_proj_nodes.append(DAG_Node(ins, o_proj_deps))

    new_nodes.extend(o_proj_nodes)

    if stop_after_op == "oproj":
        return new_nodes, o_proj_nodes

    # upgate
    upgate_instructions = schedule_upgate(globs, layer_idx)
    upgate_nodes: list[DAG_Node] = []
    for ins in upgate_instructions:
        upgate_nodes.append(DAG_Node(ins, o_proj_nodes))

    new_nodes.extend(upgate_nodes)

    if stop_after_op == "upgate":
        return new_nodes, upgate_nodes

    # downproj
    # TODO we can do better - we can start a reduction col's work once that fraction of the upgate work is done
    downproj_instructions = schedule_downproj(globs, layer_idx)
    downproj_nodes: list[DAG_Node] = []
    for ins in downproj_instructions:
        downproj_nodes.append(DAG_Node(ins, upgate_nodes))

    new_nodes.extend(downproj_nodes)

    if stop_after_op == "downproj":
        return new_nodes, downproj_nodes

    assert stop_after_op is None

    return new_nodes, downproj_nodes


class LatencyScheduleBuilder(ScheduleBuilder):
    @classmethod
    def make_globals(cls, model, seq_len: int = 0):
        return make_globals(model, seq_len=seq_len)

    @classmethod
    def make_dag(
        cls, globs, stop_after_op: str | None = None, layer_limit: int | None = None
    ):
        return make_dag(globs, stop_after_op, layer_limit)
