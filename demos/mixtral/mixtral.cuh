#pragma once

#include "kittens.cuh"
#include "megakernel.cuh"
#include <iostream>

// ---------------------------------------------------------------------------
// Opcode defines
// ---------------------------------------------------------------------------
#define OPCODE_QKV                  1
#define OPCODE_PartialAttention     2
#define OPCODE_AttentionReduction   3
#define OPCODE_OProj                4
#define OPCODE_MoE_Router           5
#define OPCODE_ExpertUpGateSiLU     6
#define OPCODE_ExpertDownProjAccum  7
#define OPCODE_RMS_LM_Head          8

// ---------------------------------------------------------------------------
// Architecture constants
// ---------------------------------------------------------------------------
#ifdef MIXTRAL_SMALL_TEST
// Small-scale dimensions for memory-constrained correctness testing.
// hidden=512, intermediate=1024, head_dim=64, 8 heads, 2 kv-heads,
// 4 experts, 2 selected. matvec_reduction=gcd(1024,512)=512.
// num_layers=1 matches the debug script's NUM_LAYERS=1 so that the lm_head
// gmem_wait index (num_layers-1=0) aligns with the Python barriers tensor.
#define MIXTRAL_NUM_LAYERS         1
#define MIXTRAL_HIDDEN_DIM         512
#define MIXTRAL_INTERMEDIATE_DIM   1024
#define MIXTRAL_HEAD_DIM           64
#define MIXTRAL_NUM_ATTN_HEADS     8
#define MIXTRAL_NUM_KV_HEADS       2
#define MIXTRAL_NUM_EXPERTS        4
#define MIXTRAL_NUM_EXPERTS_TOK    2
#define MIXTRAL_VOCAB_SIZE         256
#define MIXTRAL_KV_BLOCK_SIZE      16
#define MIXTRAL_MATVEC_BLOCK_SIZE  16
#define MIXTRAL_MATVEC_REDUCTION   512
#else
#define MIXTRAL_NUM_LAYERS         32
#define MIXTRAL_HIDDEN_DIM         4096
#define MIXTRAL_INTERMEDIATE_DIM   14336
#define MIXTRAL_HEAD_DIM           128
#define MIXTRAL_NUM_ATTN_HEADS     32
#define MIXTRAL_NUM_KV_HEADS       8
#define MIXTRAL_NUM_EXPERTS        8
#define MIXTRAL_NUM_EXPERTS_TOK    2
#define MIXTRAL_VOCAB_SIZE         32000
#define MIXTRAL_KV_BLOCK_SIZE      16
#define MIXTRAL_MATVEC_BLOCK_SIZE  16
// matvec_reduction_size = gcd(14336, 4096) = 512
#define MIXTRAL_MATVEC_REDUCTION   512
#endif
#define B200_SM_COUNT              148

constexpr int ATOMIC_ADD_START = megakernel::FREE_SLOTS_START;
constexpr int ATOMIC_ADD_END   = ATOMIC_ADD_START + 1;

using config = megakernel::default_config;

// ---------------------------------------------------------------------------
// Globals template
// ---------------------------------------------------------------------------
template <int _num_layers, int _hidden_dim, int _intermediate_dim,
          int _head_dim, int _num_attention_heads, int _num_kv_heads,
          int _num_experts, int _num_experts_per_tok,
          int _kv_block_size, int _matvec_block_size,
          int _matvec_reduction_size, int _sm_count>
struct globals_t {

    // ---------- compile-time constants ----------
    constexpr static int num_layers              = _num_layers;
    constexpr static int hidden_dim              = _hidden_dim;
    constexpr static int intermediate_dim        = _intermediate_dim;
    constexpr static int head_dim                = _head_dim;
    constexpr static int num_attention_heads     = _num_attention_heads;
    constexpr static int num_kv_heads            = _num_kv_heads;
    constexpr static int num_experts             = _num_experts;
    constexpr static int num_experts_per_tok     = _num_experts_per_tok;
    constexpr static int kv_block_size           = _kv_block_size;
    constexpr static int matvec_block_size       = _matvec_block_size;
    constexpr static int matvec_reduction_size   = _matvec_reduction_size;
    constexpr static int sm_count                = _sm_count;
    constexpr static int num_stages              = 3;

    // For downproj each col-split covers matvec_reduction_size columns.
    constexpr static int downproj_reduction_chunk_size = matvec_reduction_size;

    // QKV waits for all active-expert DownProj blocks to finish:
    //   num_experts_per_tok * (intermediate_dim/matvec_reduction_size) * (hidden_dim/matvec_block_size)
    constexpr static int qkv_expected_arrivals =
        num_experts_per_tok *
        (intermediate_dim / matvec_reduction_size) *
        (hidden_dim / matvec_block_size);

    // ---------- layout types ----------
    using instruction_layout = megakernel::instruction_layout<config>;
    using timing_layout      = megakernel::timing_layout<config>;

    // Attention/LM-head weight matrix: [1, depth, rows, hidden_dim]
    using weights_t = kittens::gl<kittens::bf16, 1, -1, -1, hidden_dim,
                                   kittens::st_bf<matvec_block_size, 512>>;

    // Wide-input weight matrix (down proj): [1, depth, rows, intermediate_dim]
    using weights_big_t = kittens::gl<kittens::bf16, 1, -1, -1, intermediate_dim,
                                       kittens::st_bf<matvec_block_size, 512>>;

    // Stacked expert gate/up weights (flattened layers*experts as depth):
    // Python tensor [num_layers*num_experts, intermediate_dim, hidden_dim]
    using expert_proj_t = kittens::gl<kittens::bf16, 1, -1, -1, hidden_dim,
                                       kittens::st_bf<matvec_block_size, 512>>;

    // Stacked expert down weights:
    // Python tensor [num_layers*num_experts, hidden_dim, intermediate_dim]
    using expert_down_t = kittens::gl<kittens::bf16, 1, -1, -1, intermediate_dim,
                                       kittens::st_bf<matvec_block_size, 512>>;

    // Router weights: [num_layers*num_experts, 1, hidden_dim]
    // Same layout as weights_t
    using router_weights_t = weights_t;

    // Activation buffers: [1, 1, 1, dim] (1D vector in last dim)
    using activations_t = kittens::gl<kittens::bf16, 1, 1, 1, hidden_dim,
                                       kittens::sv_bf<hidden_dim>,
                                       kittens::sv_bf<head_dim>,
                                       kittens::sv_bf<matvec_block_size>>;

    // Expert silu output: [1, 1, num_experts, intermediate_dim]
    // Python tensor [num_experts, intermediate_dim]
    using expert_silu_t = kittens::gl<kittens::bf16, 1, 1, -1, intermediate_dim,
                                       kittens::sv_bf<matvec_block_size>>;

    // Logits: [1, 1, 1, vocab_size] (1D, dynamic size)
    using logits_t = kittens::gl<kittens::bf16, 1, 1, 1, -1,
                                  kittens::sv_bf<matvec_block_size>>;

    // RMS norm weights: [1, 1, num_layers, hidden_dim]
    using norm_weights_t = kittens::gl<kittens::bf16, 1, 1, -1, hidden_dim,
                                        kittens::sv_bf<hidden_dim>,
                                        kittens::sv_bf<matvec_block_size>>;

    // RoPE tables: [1, 1, max_pos, head_dim]
    using rope_table_t = kittens::gl<float, 1, 1, -1, head_dim,
                                      kittens::sv_fl<head_dim>>;

    // KV cache: [num_layers, max_seq, num_kv_heads, head_dim]
    using kv_cache_t = kittens::gl<kittens::bf16, -1, -1, -1, head_dim,
                                    kittens::sv_bf<matvec_block_size>,
                                    kittens::tma::descriptor<kittens::st_bf<kv_block_size, head_dim>, 1>>;

    // Partial attention intermediates
    using attn_out_intermediates_t =
        kittens::gl<float, 1, num_attention_heads, -1, head_dim,
                    kittens::sv_fl<head_dim>>;
    using attn_lse_intermediates_t =
        kittens::gl<float, 1, 1, num_attention_heads, -1,
                    kittens::sv_fl<((sm_count + 15) / 16) * 16>>;

    // Barriers: [1, num_layers, num_opcodes, max(num_attn_heads+2*kv, num_experts)]
    using barriers = kittens::gl<uint, 1, -1, -1,
                                  num_attention_heads + 2 * num_kv_heads>;

    // Selected expert indices/scores (tiny, static size)
    using sel_indices_t = kittens::gl<int32_t, 1, 1, 1, num_experts_per_tok>;
    using sel_scores_t  = kittens::gl<kittens::bf16, 1, 1, 1, num_experts_per_tok>;

    // ---------- member fields (order must match pybind11 binding) ----------

    // Megakernel internals
    barriers         Bar;
    instruction_layout instructions;
    timing_layout    timings;

    // Attention weights
    weights_t        qkv_weights;
    norm_weights_t   attn_norm_weights;
    weights_t        o_weights;

    // MoE weights
    expert_proj_t    expert_gate_weights;
    expert_proj_t    expert_up_weights;
    expert_down_t    expert_down_weights;
    router_weights_t router_weights;
    norm_weights_t   ffn_ln_weights;

    // LM head
    norm_weights_t   lm_head_norm_weights;
    weights_t        lm_head_weights;

    // KV cache
    kv_cache_t       k_cache;
    kv_cache_t       v_cache;

    // RoPE
    rope_table_t     rope_cos;
    rope_table_t     rope_sin;

    // Activation buffers
    activations_t    hidden_states;
    activations_t    q_post_rope;
    activations_t    attn_out;
    attn_lse_intermediates_t attn_lse_intermediates;
    attn_out_intermediates_t attn_out_intermediates;
    expert_silu_t    expert_silu_out;
    activations_t    router_normed_hidden;
    logits_t         logits;

    // Router runtime outputs (plain device pointers via tiny gl)
    sel_indices_t    selected_expert_indices;
    sel_scores_t     selected_expert_scores;

    // Scalar constants
    unsigned int     pos_id;
    float            attn_scale;
    float            rms_norm_eps;
    bool             skip_attn_reduction;

    dim3 grid()  { return dim3(sm_count); }
    dim3 block() { return dim3(config::NUM_THREADS); }
    int  dynamic_shared_memory() { return config::DYNAMIC_SHARED_MEMORY; }
};

typedef globals_t<MIXTRAL_NUM_LAYERS, MIXTRAL_HIDDEN_DIM, MIXTRAL_INTERMEDIATE_DIM,
                  MIXTRAL_HEAD_DIM, MIXTRAL_NUM_ATTN_HEADS, MIXTRAL_NUM_KV_HEADS,
                  MIXTRAL_NUM_EXPERTS, MIXTRAL_NUM_EXPERTS_TOK,
                  MIXTRAL_KV_BLOCK_SIZE, MIXTRAL_MATVEC_BLOCK_SIZE,
                  MIXTRAL_MATVEC_REDUCTION, B200_SM_COUNT>
    mixtral_globals;

// Forward declarations
template <typename config = config, typename globals = mixtral_globals>
struct attention_partial;

template <typename config = config, typename globals = mixtral_globals>
struct attention_reduction;

template <typename config = config, typename globals = mixtral_globals>
struct rms_qkv_rope_append;

template <typename config = config, typename globals = mixtral_globals>
struct o_proj;

template <typename config = config, typename globals = mixtral_globals>
struct moe_router;

template <typename config = config, typename globals = mixtral_globals>
struct expert_upgate;

template <typename config = config, typename globals = mixtral_globals>
struct expert_downproj;

template <typename config = config, typename globals = mixtral_globals>
struct rms_lm_head;
