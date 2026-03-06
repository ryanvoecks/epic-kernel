"""
MK (MegaKernel) interpreter for Llama 3.2 3B latency decode.

Loads the ``mk_llama_3b`` pybind11 module built from
demos/low-latency-llama-3b/ and calls it with the same tensor interface
as the 1B interpreter.
"""

import sys
from pathlib import Path

import torch
from einops import rearrange

from megakernels.llama_3b.instructions import Globals
from megakernels.mk import MK_Interpreter


def get_mk_3b_func(mk_dir: Path):
    sys.path.append(str(mk_dir.expanduser().absolute()))
    from mk_llama_3b import mk_llama_3b  # type: ignore

    return mk_llama_3b


def interpret_with_mk(
    globs: Globals,
    mk_func,
):
    fourD_k_cache = rearrange(globs.k_cache, "l b t h d -> (l b) t h d")
    fourD_v_cache = rearrange(globs.v_cache, "l b t h d -> (l b) t h d")

    mk_func(
        # vm stuff
        globs.barriers,
        globs.instructions,
        globs.timings,
        # weights
        globs.qkv_proj_weights,
        globs.attn_ln_weights,
        globs.o_proj_weights,
        globs.mlp_ln_weights,
        globs.up_proj_weights,
        globs.gate_proj_weights,
        globs.down_proj_weights,
        globs.lm_head_norm_weights.data,
        globs.lm_head_weights.data,
        fourD_k_cache,
        fourD_v_cache,
        # rope
        globs.rope_cos,
        globs.rope_sin,
        # activations
        globs.hidden_states,
        globs.post_ln_rope_q,
        globs.attn_out,
        globs.attn_lse_intermediates,
        globs.attn_out_intermediates,
        globs.silu_out,
        globs.logits,
        # scalars
        globs.pos_id,
        globs.attn_scale,
        globs.rms_norm_eps,
        globs.skip_attn_reduction,
        stream=torch.cuda.current_stream(),
    )


class Latency3B_MK_Interpreter(MK_Interpreter):
    def __init__(self, mk_dir: Path):
        self.mk_func = get_mk_3b_func(mk_dir)

    def interpret(self, globs: Globals):
        interpret_with_mk(globs, self.mk_func)
