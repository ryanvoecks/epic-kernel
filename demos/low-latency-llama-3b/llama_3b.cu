#include "llama_3b.cuh"

#include "rms_matvec_rope_append.cu"
#include "attention_partial.cu"
#include "attention_reduction.cu"
#include "matvec_adds.cu"
#include "upgate.cu"
#include "rms_lm_head.cu"

#include "pyutils/pyutils.cuh"

using namespace kittens;
using namespace megakernel;

using rms_qkv_rope_append_op =
    rms_qkv_rope_append<default_config, llama_3b_globals>;
using attention_partial_op =
    attention_partial<default_config, llama_3b_globals>;
using attention_reduction_op =
    attention_reduction<default_config, llama_3b_globals>;
using o_proj_op = o_proj<default_config, llama_3b_globals>;
using rms_upgate_silu_op = rms_upgate_silu<default_config, llama_3b_globals>;
using downproj_op = downproj<default_config, llama_3b_globals>;
using rms_lm_head_op = rms_lm_head<default_config, llama_3b_globals>;

PYBIND11_MODULE(mk_llama_3b, m) {
    m.doc() = "";
    kittens::py::bind_kernel<
        mk<default_config, llama_3b_globals, attention_partial_op,
            attention_reduction_op, rms_qkv_rope_append_op, downproj_op,
            o_proj_op, rms_upgate_silu_op, rms_lm_head_op>>(
        m, "mk_llama_3b", &llama_3b_globals::Bar,
        &llama_3b_globals::instructions, &llama_3b_globals::timings,

        &llama_3b_globals::qkv_weights, &llama_3b_globals::attn_norm_weights,
        &llama_3b_globals::o_weights, &llama_3b_globals::mlp_norm_weights,
        &llama_3b_globals::up_weights, &llama_3b_globals::gate_weights,
        &llama_3b_globals::down_weights,
        &llama_3b_globals::lm_head_norm_weights,
        &llama_3b_globals::lm_head_weights, &llama_3b_globals::k_cache,
        &llama_3b_globals::v_cache,

        &llama_3b_globals::rope_cos, &llama_3b_globals::rope_sin,

        &llama_3b_globals::hidden_states, &llama_3b_globals::q_post_rope,
        &llama_3b_globals::attn_out, &llama_3b_globals::attn_lse_intermediates,
        &llama_3b_globals::attn_out_intermediates, &llama_3b_globals::silu_out,
        &llama_3b_globals::logits,

        &llama_3b_globals::pos_id, &llama_3b_globals::attn_scale,
        &llama_3b_globals::rms_norm_eps,
        &llama_3b_globals::skip_attn_reduction);
}
