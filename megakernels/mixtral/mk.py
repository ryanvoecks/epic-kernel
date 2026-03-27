"""MK interpreter for Mixtral — calls the compiled mk_mixtral CUDA kernel."""
import sys
from pathlib import Path

import torch
from einops import rearrange
from torch import Tensor

from megakernels.mk import MK_Interpreter
from megakernels.mixtral.instructions import MixtralGlobals


def interpret_with_mk(globs: MixtralGlobals, mk_func):
    """Call mk_mixtral with tensors in the exact order of mixtral_globals struct fields."""
    # Expert weights must be flattened from [num_layers, num_experts, ...] to
    # [num_layers * num_experts, ...] so the gl<..., 1, -1, -1, hidden_dim> can
    # index them as a single depth dimension (depth = layer * num_experts + expert).
    num_layers = globs.num_hidden_layers
    num_experts = globs.num_experts

    def flatten_expert(t: Tensor) -> Tensor:
        """[L, E, *rest] -> [L*E, *rest] contiguous."""
        L, E = t.shape[0], t.shape[1]
        return t.reshape(L * E, *t.shape[2:]).contiguous()

    expert_gate = flatten_expert(globs.expert_gate_weights)
    expert_up   = flatten_expert(globs.expert_up_weights)
    expert_down = flatten_expert(globs.expert_down_weights)

    # Router weights: [num_layers, num_experts, hidden] -> [num_layers*num_experts, hidden]
    router_w = globs.router_weights.reshape(num_layers * num_experts, -1).contiguous()

    # lm_head_norm_weights is 1D [hidden], needs to be [1, hidden] for
    # norm_weights_t = gl<bf16, 1, 1, -1, hidden_dim> (row index = layer_idx=0)
    lm_head_norm = globs.lm_head_norm_weights.unsqueeze(0)

    # KV cache: Python stores [layers, batch, seq, kv_heads, head_dim] (5D)
    # CUDA expects [layers, seq, kv_heads, head_dim] (4D, batch must be 1)
    fourD_k = rearrange(globs.k_cache, "l b t h d -> (l b) t h d")
    fourD_v = rearrange(globs.v_cache, "l b t h d -> (l b) t h d")

    mk_func(
        # Megakernel internals
        globs.barriers,
        globs.instructions,
        globs.timings,

        # Attention weights
        globs.qkv_proj_weights,      # qkv_weights
        globs.attn_ln_weights,       # attn_norm_weights
        globs.o_proj_weights,        # o_weights

        # MoE weights (flattened to 3D)
        expert_gate,                 # expert_gate_weights
        expert_up,                   # expert_up_weights
        expert_down,                 # expert_down_weights
        router_w,                    # router_weights
        globs.ffn_ln_weights,        # ffn_ln_weights

        # LM head
        lm_head_norm,                # lm_head_norm_weights
        globs.lm_head_weights.data,  # lm_head_weights (nn.Parameter → plain Tensor for C++ binding)

        # KV cache (4D for CUDA)
        fourD_k,
        fourD_v,

        # RoPE
        globs.rope_cos,
        globs.rope_sin,

        # Activation buffers
        globs.hidden_states,
        globs.post_ln_rope_q,        # q_post_rope
        globs.attn_out,
        globs.attn_lse_intermediates,
        globs.attn_out_intermediates,
        globs.expert_silu_out,
        globs.router_normed_hidden,
        globs.logits,

        # Router runtime outputs
        globs.selected_expert_indices,
        globs.selected_expert_scores,

        # Speculative expert prediction
        globs.predicted_expert_indices,

        # Scalar constants
        globs.pos_id,
        globs.attn_scale,
        globs.rms_norm_eps,
        globs.skip_attn_reduction,

        stream=torch.cuda.current_stream(),
    )


class MixtralMK_Interpreter(MK_Interpreter):
    def __init__(self, mk_dir: Path):
        sys.path.append(str(mk_dir.expanduser().absolute()))
        from mk_mixtral import mk_mixtral  # type: ignore

        self.mk_func = mk_mixtral

    def interpret(self, globs: MixtralGlobals):
        interpret_with_mk(globs, self.mk_func)
