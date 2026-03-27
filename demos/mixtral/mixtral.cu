#include "mixtral.cuh"
#include "utils.cuh"
#include "matvec_pipeline.cuh"

#include "rms_matvec_rope_append.cu"
#include "attention_partial.cu"
#include "attention_reduction.cu"
#include "matvec_adds.cu"
#include "rms_router_matvec.cu"
#include "expert_downproj.cu"
#include "rms_lm_head.cu"

#include "pyutils/pyutils.cuh"

using namespace kittens;
using namespace megakernel;

using rms_qkv_rope_append_op    = rms_qkv_rope_append<config, mixtral_globals>;
using attention_partial_op      = attention_partial<config, mixtral_globals>;
using attention_reduction_op    = attention_reduction<config, mixtral_globals>;
using o_proj_op                 = o_proj<config, mixtral_globals>;
using rms_router_upgate_op      = rms_router_upgate<config, mixtral_globals>;
using expert_downproj_fused_op  = expert_downproj_fused<config, mixtral_globals>;
using rms_lm_head_op            = rms_lm_head<config, mixtral_globals>;

PYBIND11_MODULE(mk_mixtral, m)
{
    m.doc() = "";
    kittens::py::bind_kernel<
        mk<config, mixtral_globals,
           rms_qkv_rope_append_op,
           attention_partial_op,
           attention_reduction_op,
           o_proj_op,
           rms_router_upgate_op,
           expert_downproj_fused_op,
           rms_lm_head_op>>(
        m, "mk_mixtral",

        // Megakernel internals (in struct field order)
        &mixtral_globals::Bar,
        &mixtral_globals::instructions,
        &mixtral_globals::timings,

        // Attention weights
        &mixtral_globals::qkv_weights,
        &mixtral_globals::attn_norm_weights,
        &mixtral_globals::o_weights,

        // MoE weights
        &mixtral_globals::expert_gate_weights,
        &mixtral_globals::expert_up_weights,
        &mixtral_globals::expert_down_weights,
        &mixtral_globals::router_weights,
        &mixtral_globals::ffn_ln_weights,

        // LM head
        &mixtral_globals::lm_head_norm_weights,
        &mixtral_globals::lm_head_weights,

        // KV cache
        &mixtral_globals::k_cache,
        &mixtral_globals::v_cache,

        // RoPE
        &mixtral_globals::rope_cos,
        &mixtral_globals::rope_sin,

        // Activation buffers
        &mixtral_globals::hidden_states,
        &mixtral_globals::q_post_rope,
        &mixtral_globals::attn_out,
        &mixtral_globals::attn_lse_intermediates,
        &mixtral_globals::attn_out_intermediates,
        &mixtral_globals::expert_silu_out,
        &mixtral_globals::router_normed_hidden,
        &mixtral_globals::logits,

        // Router runtime outputs
        &mixtral_globals::selected_expert_indices,
        &mixtral_globals::selected_expert_scores,

        // Scalar constants
        &mixtral_globals::pos_id,
        &mixtral_globals::attn_scale,
        &mixtral_globals::rms_norm_eps,
        &mixtral_globals::skip_attn_reduction);
}
