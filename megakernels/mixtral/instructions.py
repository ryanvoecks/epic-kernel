"""
Mixtral megakernel instructions and globals.

Opcode mapping:
  1 - Mixtral_QKV            (LayerNorm + QKV matvec + RoPE + KV append)
  2 - Mixtral_PartialAttn    (Partial flash attention)
  3 - Mixtral_AttnReduction  (Tree-reduce partial attention)
  4 - Mixtral_OProj          (Output projection + residual)
  5 - RmsRouterUpgate        (RMS norm + router + upgate+SiLU for both selected experts)
  6 - ExpertDownProjFused    (downproj for both selected experts)
  7 - Mixtral_RMS_LM_Head    (Final RMS norm + LM head matvec)
"""
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from megakernels.instructions import BaseGlobals, Instruction


@dataclass
class MixtralGlobals(BaseGlobals):
    # Attention activation buffers (same as latency Globals)
    post_ln_rope_q: Tensor
    attn_out: Tensor
    attn_lse_intermediates: Tensor
    attn_out_intermediates: Tensor

    # MoE activations
    expert_silu_out: Tensor          # [num_experts, intermediate_size]
    router_normed_hidden: Tensor     # [hidden_size] — RMS-normed hidden, written by router, read by experts
    logits: Tensor                   # [vocab_size]

    # Router runtime outputs (written by RmsRouterUpgate, read by ExpertDownProjFused)
    selected_expert_indices: Tensor  # [num_experts_per_tok] int32
    selected_expert_scores: Tensor   # [num_experts_per_tok] bfloat16

    # MoE stacked weights
    router_weights: Tensor           # [num_layers, num_experts, hidden_size]
    expert_gate_weights: Tensor      # [num_layers, num_experts, intermediate_size, hidden_size]
    expert_up_weights: Tensor        # [num_layers, num_experts, intermediate_size, hidden_size]
    expert_down_weights: Tensor      # [num_layers, num_experts, hidden_size, intermediate_size]
    ffn_ln_weights: Tensor           # [num_layers, hidden_size]  (pre-MoE layernorm)

    # MoE constants
    num_experts: int
    num_experts_per_tok: int

    # Attention mode flag
    skip_attn_reduction: bool

    # Block size constants
    expert_proj_block_size: int
    down_proj_block_size: int
    o_proj_block_size: int
    lm_head_block_size: int
    matvec_reduction_size: int
    qkv_block_size: int
    attn_kv_block_size: int
    attn_reduction_size: int         # GQA ratio

    def __post_init__(self):
        super().__post_init__()

    def num_total_heads(self) -> int:
        return self.num_attention_heads + self.num_kv_heads * 2


# ---------------------------------------------------------------------------
# Opcode 1 — QKV
# ---------------------------------------------------------------------------

@dataclass
class Mixtral_QKV(Instruction):
    """RMS norm + QKV matvec + interleaved RoPE + KV cache append."""
    layer_idx: int
    start_output_block_idx: int
    end_output_block_idx: int

    @classmethod
    def opcode(cls) -> int:
        return 1

    @classmethod
    def prev_opcode(cls) -> int:
        # Cross-layer dependency: previous layer's ExpertDownProjFused (opcode 6)
        return ExpertDownProjFused.opcode()

    def block_indices(self):
        return list(range(self.start_output_block_idx, self.end_output_block_idx))

    def cost(self, globs: MixtralGlobals) -> float:
        return (
            (self.end_output_block_idx - self.start_output_block_idx)
            * globs.qkv_block_size
            * globs.hidden_size
        )


# ---------------------------------------------------------------------------
# Opcode 2 — Partial Attention
# ---------------------------------------------------------------------------

@dataclass
class Mixtral_PartialAttn(Instruction):
    layer_idx: int
    kv_head_idx: int
    num_partials: int
    partial_idx: int

    @classmethod
    def opcode(cls) -> int:
        return 2

    @classmethod
    def prev_opcode(cls) -> int:
        return Mixtral_QKV.opcode()

    def cost(self, globs: MixtralGlobals) -> float:
        seq_len = globs.pos_id + 1
        return (seq_len / self.num_partials) * globs.head_dim * 2


# ---------------------------------------------------------------------------
# Opcode 3 — Attention Reduction
# ---------------------------------------------------------------------------

@dataclass
class Mixtral_AttnReduction(Instruction):
    layer_idx: int
    head_start_idx: int
    num_partials: int
    is_terminal: bool
    reduction_list: list[int]
    output_partial_idx: Optional[int] = None

    @classmethod
    def opcode(cls) -> int:
        return 3

    @classmethod
    def prev_opcode(cls) -> int:
        return Mixtral_PartialAttn.opcode()

    def cost(self, globs: MixtralGlobals) -> float:
        gqa_ratio = globs.num_attention_heads // globs.num_kv_heads
        return gqa_ratio * self.num_partials * globs.head_dim


# ---------------------------------------------------------------------------
# Opcode 4 — Output Projection + Residual
# ---------------------------------------------------------------------------

@dataclass
class Mixtral_OProj(Instruction):
    layer_idx: int
    start_block_idx: int
    end_block_idx: int
    reduction_block_idx: int

    @classmethod
    def opcode(cls) -> int:
        return 4

    @classmethod
    def prev_opcode(cls) -> int:
        return Mixtral_AttnReduction.opcode()

    def cost(self, globs: MixtralGlobals) -> float:
        return (
            (self.end_block_idx - self.start_block_idx)
            * globs.o_proj_block_size
            * globs.hidden_size
        )


# ---------------------------------------------------------------------------
# Opcode 5 — RmsRouterUpgate (fused RMS norm + router + upgate+SiLU)
# ---------------------------------------------------------------------------

@dataclass
class RmsRouterUpgate(Instruction):
    """
    Fused: RMS norm + router top-2 selection + gate+up matvec + SiLU
    for both selected experts. Each instruction handles a contiguous
    range of output blocks.
    """
    layer_idx: int
    start_block_idx: int
    num_blocks: int

    @classmethod
    def opcode(cls) -> int:
        return 5

    @classmethod
    def prev_opcode(cls) -> int:
        return Mixtral_OProj.opcode()

    def block_indices(self) -> list[int]:
        return list(range(self.start_block_idx, self.start_block_idx + self.num_blocks))

    def cost(self, globs: MixtralGlobals) -> float:
        # Router cost + 2 experts * gate+up matvec
        return (
            globs.num_experts * globs.hidden_size
            + self.num_blocks * globs.expert_proj_block_size * globs.hidden_size * 2 * globs.num_experts_per_tok
        )


# ---------------------------------------------------------------------------
# Opcode 6 — ExpertDownProjFused (downproj for both selected experts)
# ---------------------------------------------------------------------------

@dataclass
class ExpertDownProjFused(Instruction):
    """
    Down proj for both selected experts + weighted atomic-add into hidden_states.
    Each instruction handles one reduction column split across all output blocks.
    """
    layer_idx: int
    start_block_idx: int
    end_block_idx: int
    reduction_block_idx: int

    @classmethod
    def opcode(cls) -> int:
        return 6

    @classmethod
    def prev_opcode(cls) -> int:
        return RmsRouterUpgate.opcode()

    def cost(self, globs: MixtralGlobals) -> float:
        return (
            (self.end_block_idx - self.start_block_idx)
            * globs.down_proj_block_size
            * globs.intermediate_size
            * globs.num_experts_per_tok
        )


# ---------------------------------------------------------------------------
# Opcode 7 — RMS norm + LM Head
# ---------------------------------------------------------------------------

@dataclass
class Mixtral_RMS_LM_Head(Instruction):
    start_output_block_idx: int
    end_output_block_idx: int

    @classmethod
    def opcode(cls) -> int:
        return 7

    @classmethod
    def prev_opcode(cls) -> int:
        return ExpertDownProjFused.opcode()

    def cost(self, globs: MixtralGlobals) -> float:
        return (
            (self.end_output_block_idx - self.start_output_block_idx)
            * globs.lm_head_block_size
            * globs.hidden_size
        )
