"""
Python VM (reference interpreter) for Mixtral megakernel instructions.

Executes each instruction sequentially, matching the semantics the CUDA
kernel must implement.  Used for correctness testing (diff vs. HF Mixtral).
"""
import math

import torch
import torch.nn.functional as F
from einops import einsum
from torch import Tensor

from megakernels.demos.latency.python_vm import (
    get_start_end,
    matvec,
    matvec_with_residual,  # used by solve_oproj
    rms_norm,
)
from megakernels.llama import apply_rotary_pos_emb_interleaved
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


# ---------------------------------------------------------------------------
# Barrier helpers
# ---------------------------------------------------------------------------

def _blocks_per_head(globs: MixtralGlobals) -> int:
    return globs.head_dim // globs.qkv_block_size


def _expected_oproj_barrier(globs: MixtralGlobals) -> int:
    return globs.hidden_size // globs.o_proj_block_size


def _expected_upgate_barrier(globs: MixtralGlobals) -> int:
    """Expected sum of UpGate barriers for ONE expert."""
    return globs.intermediate_size // globs.expert_proj_block_size


def _expected_downproj_barrier(globs: MixtralGlobals) -> int:
    """Expected total DownProj increments for all active experts combined."""
    num_col_splits = globs.intermediate_size // globs.matvec_reduction_size
    num_down_blocks = globs.hidden_size // globs.down_proj_block_size
    return globs.num_experts_per_tok * num_col_splits * num_down_blocks


# ---------------------------------------------------------------------------
# Opcode 1 — Mixtral_QKV
# ---------------------------------------------------------------------------

def solve_qkv(globs: MixtralGlobals, ins: Mixtral_QKV):
    layer_idx = ins.layer_idx
    if layer_idx > 0:
        op_barriers = globs.barriers[layer_idx - 1, ins.prev_opcode() - 1]
        assert op_barriers[0] == _expected_downproj_barrier(globs), (
            f"Layer {layer_idx}: expected downproj barrier {_expected_downproj_barrier(globs)}, "
            f"got {op_barriers[0]}"
        )

    post_ln = rms_norm(
        inp=globs.hidden_states,
        weight=globs.attn_ln_weights[layer_idx],
        eps=globs.rms_norm_eps,
    )

    pos_id = globs.pos_id
    k_start = globs.num_attention_heads * globs.head_dim
    v_start = k_start + globs.num_kv_heads * globs.head_dim

    barriers = globs.barriers[layer_idx, ins.opcode() - 1]
    bph = _blocks_per_head(globs)

    for block_idx in range(ins.start_output_block_idx, ins.end_output_block_idx):
        start, end = get_start_end(globs.qkv_block_size, block_idx)

        if start < k_start:
            mode = "q"
        elif start < v_start:
            mode = "k"
        else:
            mode = "v"

        out = einsum(
            globs.qkv_proj_weights[layer_idx][start:end],
            post_ln,
            "o i, i -> o",
        )

        if mode in ("q", "k"):
            full_head = torch.zeros(1, globs.head_dim, device=globs.hidden_states.device, dtype=out.dtype)
            seg = start % globs.head_dim
            full_head[:, seg : seg + (end - start)] = out
            full_head_rope, _ = apply_rotary_pos_emb_interleaved(
                q=full_head, k=full_head,
                cos=globs.rope_cos[pos_id],
                sin=globs.rope_sin[pos_id],
                unsqueeze_dim=0,
            )
            out = full_head_rope[:, seg : seg + (end - start)].view(-1)

        match mode:
            case "q":
                globs.post_ln_rope_q[start:end] = out
            case "k":
                s_in_k = start - k_start
                globs.k_cache[layer_idx, :, pos_id].view(-1)[s_in_k : s_in_k + (end - start)] = out
            case "v":
                s_in_v = start - v_start
                globs.v_cache[layer_idx, :, pos_id].view(-1)[s_in_v : s_in_v + (end - start)] = out

        barriers[block_idx // bph] += 1


# ---------------------------------------------------------------------------
# Opcode 2 — Mixtral_PartialAttn
# ---------------------------------------------------------------------------

def solve_partial_attn(globs: MixtralGlobals, ins: Mixtral_PartialAttn):
    gqa_ratio = globs.num_attention_heads // globs.num_kv_heads
    bph = _blocks_per_head(globs)

    op_barriers = globs.barriers[ins.layer_idx, ins.prev_opcode() - 1]
    for i in range(gqa_ratio):
        assert op_barriers[ins.kv_head_idx * gqa_ratio + i] == bph
    assert op_barriers[globs.num_attention_heads + ins.kv_head_idx] == bph
    assert op_barriers[globs.num_attention_heads + globs.num_kv_heads + ins.kv_head_idx] == bph

    kv_block_size = globs.attn_kv_block_size
    seq_len = globs.pos_id + 1
    layer_idx = ins.layer_idx
    kv_head_idx = ins.kv_head_idx

    total_blocks = math.ceil(seq_len / kv_block_size)
    blocks_per_partial = math.ceil(total_blocks / ins.num_partials)
    start_block = ins.partial_idx * blocks_per_partial
    end_block = min(start_block + blocks_per_partial, total_blocks)
    start_token = start_block * kv_block_size
    end_token = min(end_block * kv_block_size, seq_len)

    k = globs.k_cache[layer_idx, 0, start_token:end_token, kv_head_idx]
    v = globs.v_cache[layer_idx, 0, start_token:end_token, kv_head_idx]

    head_start = kv_head_idx * gqa_ratio
    head_end = head_start + gqa_ratio
    q = globs.post_ln_rope_q.view(globs.num_attention_heads, -1)[head_start:head_end]

    qk = einsum(q.float(), k.float(), "h i, k i -> h k") * globs.attn_scale
    lse = torch.log2(torch.sum(torch.exp(qk), dim=-1))
    out = einsum(torch.softmax(qk, dim=-1), v.float(), "h k, k o -> h o")

    if globs.skip_attn_reduction:
        globs.attn_out.view(globs.num_attention_heads, -1)[head_start:head_end] = out
        globs.barriers[layer_idx, Mixtral_AttnReduction.opcode() - 1][0] += head_end - head_start
    else:
        globs.attn_lse_intermediates[head_start:head_end, ins.partial_idx] = lse
        globs.attn_out_intermediates[head_start:head_end, ins.partial_idx] = out
        globs.barriers[layer_idx, ins.opcode() - 1][head_start:head_end] += 1


# ---------------------------------------------------------------------------
# Opcode 3 — Mixtral_AttnReduction
# ---------------------------------------------------------------------------

def solve_attn_reduction(globs: MixtralGlobals, ins: Mixtral_AttnReduction):
    head_start = ins.head_start_idx

    op_barriers = globs.barriers[ins.layer_idx, ins.prev_opcode() - 1]
    assert op_barriers[head_start] == ins.num_partials

    indices = torch.tensor(ins.reduction_list, dtype=torch.long, device=globs.hidden_states.device)
    lses = globs.attn_lse_intermediates[head_start : head_start + globs.attn_reduction_size, indices]
    outs = globs.attn_out_intermediates[head_start : head_start + globs.attn_reduction_size, indices]

    max_lse = torch.max(lses, dim=-1, keepdim=True).values
    factors = (lses - max_lse).exp2()
    denom = factors.sum(dim=-1, keepdim=True)
    reduced = (outs * factors.unsqueeze(-1)).sum(dim=1) / denom

    if ins.is_terminal:
        globs.attn_out.view(globs.num_attention_heads, -1)[head_start : head_start + globs.attn_reduction_size] = reduced
    else:
        slot = ins.output_partial_idx
        globs.attn_lse_intermediates[head_start : head_start + globs.attn_reduction_size, slot] = denom.log()
        globs.attn_out_intermediates[head_start : head_start + globs.attn_reduction_size, slot] = reduced

    globs.barriers[ins.layer_idx, ins.opcode() - 1][0] += globs.attn_reduction_size


# ---------------------------------------------------------------------------
# Opcode 4 — Mixtral_OProj
# ---------------------------------------------------------------------------

def solve_oproj(globs: MixtralGlobals, ins: Mixtral_OProj):
    op_barriers = globs.barriers[ins.layer_idx, ins.prev_opcode() - 1]
    assert op_barriers[0] == globs.num_attention_heads

    assert ins.start_block_idx == ins.end_block_idx - 1
    assert ins.reduction_block_idx == 0

    matvec_with_residual(
        mat=globs.o_proj_weights[ins.layer_idx],
        vec=globs.attn_out,
        residual=globs.hidden_states,
        block_size=globs.o_proj_block_size,
        start_block_idx=ins.start_block_idx,
        end_block_idx=ins.end_block_idx,
        reduction_size=globs.matvec_reduction_size,
        reduction_block_idx=ins.reduction_block_idx,
    )

    globs.barriers[ins.layer_idx, ins.opcode() - 1][0] += ins.end_block_idx - ins.start_block_idx


# ---------------------------------------------------------------------------
# Opcode 5 — MoE_Router
# ---------------------------------------------------------------------------

def solve_router(globs: MixtralGlobals, ins: MoE_Router):
    op_barriers = globs.barriers[ins.layer_idx, ins.prev_opcode() - 1]
    assert op_barriers[0] == _expected_oproj_barrier(globs), (
        f"Router: expected oproj barrier {_expected_oproj_barrier(globs)}, got {op_barriers[0]}"
    )

    # 1. RMS norm of hidden_states (FFN pre-norm)
    normed = rms_norm(
        inp=globs.hidden_states,
        weight=globs.ffn_ln_weights[ins.layer_idx],
        eps=globs.rms_norm_eps,
    )

    # 2. Router matVec: [num_experts, hidden] @ [hidden] → [num_experts]
    router_logits = (globs.router_weights[ins.layer_idx].float() @ normed.float())

    # 3. Softmax + top-k
    scores = torch.softmax(router_logits, dim=-1)
    top_scores, top_indices = torch.topk(scores, globs.num_experts_per_tok)
    top_scores = top_scores / top_scores.sum()  # renormalize

    globs.selected_expert_indices.copy_(top_indices.to(torch.int32))
    globs.selected_expert_scores.copy_(top_scores.to(globs.selected_expert_scores.dtype))

    # Save normed hidden for reuse by all ExpertUpGateSiLU instructions
    globs.router_normed_hidden = normed

    # Signal that router is complete
    globs.barriers[ins.layer_idx, ins.opcode() - 1][0] = 1


# ---------------------------------------------------------------------------
# Opcode 6 — ExpertUpGateSiLU
# ---------------------------------------------------------------------------

def solve_expert_upgate(globs: MixtralGlobals, ins: ExpertUpGateSiLU):
    # Check router is done
    op_barriers = globs.barriers[ins.layer_idx, ins.prev_opcode() - 1]
    assert op_barriers[0] == 1, (
        f"ExpertUpGate: router barrier not set (layer={ins.layer_idx})"
    )

    # Skip if this expert was not selected
    active_experts = globs.selected_expert_indices.tolist()
    if ins.expert_idx not in active_experts:
        # Still increment barrier so ExpertDownProj can proceed
        globs.barriers[ins.layer_idx, ins.opcode() - 1][ins.expert_idx] += ins.num_blocks
        return

    assert globs.router_normed_hidden is not None
    normed = globs.router_normed_hidden

    block_size = globs.expert_proj_block_size
    for block_idx in ins.block_indices():
        start, end = get_start_end(block_size, block_idx)
        gate_out, _, _ = matvec(
            mat=globs.expert_gate_weights[ins.layer_idx, ins.expert_idx],
            vec=normed,
            block_size=block_size,
            block_idx=block_idx,
        )
        up_out, _, _ = matvec(
            mat=globs.expert_up_weights[ins.layer_idx, ins.expert_idx],
            vec=normed,
            block_size=block_size,
            block_idx=block_idx,
        )
        globs.expert_silu_out[ins.expert_idx, start:end] = (
            F.silu(gate_out) * up_out
        ).to(globs.expert_silu_out.dtype)

    globs.barriers[ins.layer_idx, ins.opcode() - 1][ins.expert_idx] += ins.num_blocks


# ---------------------------------------------------------------------------
# Opcode 7 — ExpertDownProjAccum
# ---------------------------------------------------------------------------

def solve_expert_downproj(globs: MixtralGlobals, ins: ExpertDownProjAccum):
    # Check UpGate for THIS expert is done
    op_barriers = globs.barriers[ins.layer_idx, ins.prev_opcode() - 1]
    expected_upgate = _expected_upgate_barrier(globs)
    assert op_barriers[ins.expert_idx] == expected_upgate, (
        f"ExpertDownProj: expert {ins.expert_idx} upgate barrier "
        f"expected {expected_upgate}, got {op_barriers[ins.expert_idx]}"
    )

    # Skip if this expert was not selected
    active_experts = globs.selected_expert_indices.tolist()
    if ins.expert_idx not in active_experts:
        return

    # Look up weight for this expert
    slot = active_experts.index(ins.expert_idx)
    weight = globs.selected_expert_scores[slot].float()

    # Weighted accumulate into hidden_states
    block_size = globs.down_proj_block_size
    for block_idx in range(ins.start_block_idx, ins.end_block_idx):
        out, start, end = matvec(
            mat=globs.expert_down_weights[ins.layer_idx, ins.expert_idx],
            vec=globs.expert_silu_out[ins.expert_idx],
            block_size=block_size,
            block_idx=block_idx,
            reduce=True,
            reduction_size=globs.matvec_reduction_size,
            reduction_idx=ins.reduction_block_idx,
        )
        globs.hidden_states[start:end] += (weight * out).to(globs.hidden_states.dtype)

    globs.barriers[ins.layer_idx, ins.opcode() - 1][0] += (
        ins.end_block_idx - ins.start_block_idx
    )


# ---------------------------------------------------------------------------
# Opcode 8 — Mixtral_RMS_LM_Head
# ---------------------------------------------------------------------------

def solve_lm_head(globs: MixtralGlobals, ins: Mixtral_RMS_LM_Head):
    op_barriers = globs.barriers[globs.num_hidden_layers - 1, ins.prev_opcode() - 1]
    assert op_barriers[0] == _expected_downproj_barrier(globs), (
        f"LM head: expected barrier {_expected_downproj_barrier(globs)}, got {op_barriers[0]}"
    )

    post_ln = rms_norm(
        inp=globs.hidden_states,
        weight=globs.lm_head_norm_weights,
        eps=globs.rms_norm_eps,
    )

    block_size = globs.lm_head_block_size
    for block_idx in range(ins.start_output_block_idx, ins.end_output_block_idx):
        start, end = get_start_end(block_size, block_idx)
        out = einsum(globs.lm_head_weights[start:end], post_ln, "o i, i -> o")
        globs.logits[start:end] = out


# ---------------------------------------------------------------------------
# Instruction → Solver map
# ---------------------------------------------------------------------------

INSTRUCTION_TO_SOLVER = {
    Mixtral_QKV: solve_qkv,
    Mixtral_PartialAttn: solve_partial_attn,
    Mixtral_AttnReduction: solve_attn_reduction,
    Mixtral_OProj: solve_oproj,
    MoE_Router: solve_router,
    ExpertUpGateSiLU: solve_expert_upgate,
    ExpertDownProjAccum: solve_expert_downproj,
    Mixtral_RMS_LM_Head: solve_lm_head,
}
