// Partial attention instruction, followed by attention reduction

#include "llama.cuh"

using namespace kittens;
using namespace megakernel;

template <typename config, typename globals> struct attention_partial {
    static constexpr int opcode = OPCODE_PartialAttention;
    // NUM_STAGES is model-dependent
    static constexpr int NUM_STAGES = globals::num_stages;
    static constexpr int GQA_RATIO =
        globals::num_attention_heads / globals::num_kv_heads;
    static constexpr int QOL_PAGE = 0;
    static constexpr int KV_PAGE = 1;

    using q_rt = kittens::rt_bf<16, globals::head_dim>;
    using q_st = kittens::st_bf<16, globals::head_dim>;
    using k_rt = kittens::rt_bf<globals::kv_block_size, globals::head_dim>;
    using v_rt = kittens::rt_bf<globals::kv_block_size, globals::head_dim, col_l>;
    using kv_st = kittens::st_bf<globals::kv_block_size, globals::head_dim>;
    using attn_fl_rt = kittens::rt_fl<16, globals::kv_block_size>;
    using attn_bf_rt = kittens::rt_bf<16, globals::kv_block_size>;
    using max_vec_rv = col_vec<kittens::rt_fl<16, globals::head_dim>>;
    using max_vec_sv = kittens::sv_fl<16>;
    using norm_vec_rv = col_vec<kittens::rt_fl<16, globals::head_dim>>;
    using norm_vec_sv = kittens::sv_fl<16>;
    using l_rv = col_vec<kittens::rt_fl<16, globals::head_dim>>;
    using l_sv = kittens::sv_fl<16>;
    using o_rt = kittens::rt_fl<16, globals::head_dim>;
    using o_sv = kittens::sv_fl<globals::head_dim>;
    using o_sv_bf = kittens::sv_bf<globals::head_dim>;

    struct parsed_instruction {
        int layer_idx;
        int kv_head_idx;
        int num_partials;
        int partial_idx;
        __device__ inline parsed_instruction(
            typename config::instruction_t &instruction) {
            layer_idx = instruction[1];
            kv_head_idx = instruction[2];
            num_partials = instruction[3];
            partial_idx = instruction[4];
        }
        __device__ inline parsed_instruction(megakernel::state<config> &s)
            : parsed_instruction(s.instruction()) {}
    };

    // Semaphores
    __device__ static inline kittens::semaphore &Q_arrived(megakernel::state<config> &s) {
        return s.semaphores()[0];
    }
    __device__ static inline kittens::semaphore &O_arrived(megakernel::state<config> &s) {
        return s.semaphores()[1];
    }
    __device__ static inline kittens::semaphore &L_arrived(megakernel::state<config> &s) {
        return s.semaphores()[2];
    }
    __device__ static inline kittens::semaphore &K_arrived(megakernel::state<config> &s, int stage) {
        return s.semaphores()[3 + stage * 2];
    }
    __device__ static inline kittens::semaphore &V_arrived(megakernel::state<config> &s, int stage) {
        return s.semaphores()[3 + stage * 2 + 1];
    }
    __device__ static inline kittens::semaphore &K_finished(megakernel::state<config> &s, int stage) {
        return s.semaphores()[3 + NUM_STAGES * 2 + stage * 2];
    }
    __device__ static inline kittens::semaphore &V_finished(megakernel::state<config> &s, int stage) {
        return s.semaphores()[3 + NUM_STAGES * 2 + stage * 2 + 1];
    }

    __device__ static inline void wait_QOL_page(megakernel::state<config> &s) {
        s.wait_page_ready(s.pid(QOL_PAGE));
    }
    __device__ static inline void wait_KV_page(megakernel::state<config> &s) {
        s.wait_page_ready(s.pid(KV_PAGE));
    }
    __device__ static inline void finish_QOL_page(megakernel::state<config> &s) {
        if (kittens::warp::laneid() == 0)
            s.finish_page(s.pid(QOL_PAGE), config::NUM_CONSUMER_WARPS);
    }
    __device__ static inline void finish_KV_page(megakernel::state<config> &s) {
        if (kittens::warp::laneid() == 0)
            s.finish_page(s.pid(KV_PAGE), config::NUM_CONSUMER_WARPS);
    }

    // QOL page layout:
    //   q_st: st_bf<16,head_dim> = head_dim*16*2 bytes
    //   o_sv[GQA_RATIO]: GQA_RATIO × sv_fl<head_dim>
    //   l_sv: sv_fl<16>
    __device__ static inline q_st &get_Q_smem(megakernel::state<config> &s) {
        int pid = s.pid(QOL_PAGE);
        return *reinterpret_cast<q_st *>(s.pages[pid].data);
    }
    __device__ static inline o_sv (&get_O_smem(megakernel::state<config> &s))[GQA_RATIO] {
        int pid = s.pid(QOL_PAGE);
        return *reinterpret_cast<o_sv(*)[GQA_RATIO]>(
            reinterpret_cast<char *>(s.pages[pid].data) + sizeof(q_st));
    }
    __device__ static inline l_sv &get_L_smem(megakernel::state<config> &s) {
        int pid = s.pid(QOL_PAGE);
        return *reinterpret_cast<l_sv *>(
            reinterpret_cast<char *>(s.pages[pid].data) + sizeof(q_st) +
            sizeof(o_sv) * GQA_RATIO);
    }
    // KV page layout: NUM_STAGES × (K + V) tiles
    __device__ static inline kv_st &get_K_smem(megakernel::state<config> &s, int stage) {
        int pid = s.pid(KV_PAGE);
        return *reinterpret_cast<kv_st *>(
            reinterpret_cast<char *>(s.pages[pid].data) +
            sizeof(kv_st) * (stage * 2));
    }
    __device__ static inline kv_st &get_V_smem(megakernel::state<config> &s, int stage) {
        int pid = s.pid(KV_PAGE);
        return *reinterpret_cast<kv_st *>(
            reinterpret_cast<char *>(s.pages[pid].data) +
            sizeof(kv_st) * (1 + stage * 2));
    }

    // Store GQA_RATIO rows from a 16-row register tile into GQA_RATIO separate sv_fl<head_dim> vectors.
    template <ducks::sv::all SV, ducks::rt::all RT>
    __device__ static inline void
    store_gqa_rows(SV (&dst)[GQA_RATIO], const RT &src, int row3idx /*= 0, 1, or 2*/) {
        static_assert(RT::rows == 16, "src rows must be 16.");
        static_assert(SV::length == src.cols, "dst length must match src cols.");

        using T2 = RT::dtype;
        using T = base_types::packing<T2>::unpacked_type;
        using U = SV::dtype;
        using U2 = base_types::packing<U>::packed_type;

        uint32_t dst_ptr[GQA_RATIO];
        for (int i = 0; i < GQA_RATIO; i++) {
            dst_ptr[i] = static_cast<uint32_t>(__cvta_generic_to_shared(&dst[i].data[0]));
        }

        int laneid = kittens::warp::laneid();
        int local_row_idx = (laneid % 16) / 4;
        int local_col_idx = laneid % 4;

        if (row3idx % 2 == 0 && laneid < 16) {
            if (row3idx / 2 == 0) {  // rows 0-3
                for (int j = 0; j < src.width; j++) {
                    U2 tmp[2];
                    tmp[0] = base_types::convertor<U2, T2>::convert(src.tiles[0][j].data[0]);
                    tmp[1] = base_types::convertor<U2, T2>::convert(src.tiles[0][j].data[2]);
                    int col_idx = local_col_idx * 2 + j * 16;
                    if (local_row_idx < GQA_RATIO) {
                        move<U2>::sts(dst_ptr[local_row_idx] + sizeof(U) * col_idx, tmp[0]);
                        move<U2>::sts(dst_ptr[local_row_idx] + sizeof(U) * (col_idx + 8), tmp[1]);
                    }
                }
            } else {  // rows 8-11
                for (int j = 0; j < src.width; j++) {
                    U2 tmp[2];
                    tmp[0] = base_types::convertor<U2, T2>::convert(src.tiles[0][j].data[1]);
                    tmp[1] = base_types::convertor<U2, T2>::convert(src.tiles[0][j].data[3]);
                    int col_idx = local_col_idx * 2 + j * 16;
                    if (local_row_idx < GQA_RATIO) {
                        move<U2>::sts(dst_ptr[local_row_idx] + sizeof(U) * col_idx, tmp[0]);
                        move<U2>::sts(dst_ptr[local_row_idx] + sizeof(U) * (col_idx + 8), tmp[1]);
                    }
                }
            }
        } else if (row3idx % 2 == 1 && laneid >= 16) {  // rows 4-7
            if (row3idx / 2 == 0) {
                for (int j = 0; j < src.width; j++) {
                    U2 tmp[2];
                    tmp[0] = base_types::convertor<U2, T2>::convert(src.tiles[0][j].data[0]);
                    tmp[1] = base_types::convertor<U2, T2>::convert(src.tiles[0][j].data[2]);
                    int col_idx = local_col_idx * 2 + j * 16;
                    if (local_row_idx < GQA_RATIO) {
                        move<U2>::sts(dst_ptr[local_row_idx] + sizeof(U) * col_idx, tmp[0]);
                        move<U2>::sts(dst_ptr[local_row_idx] + sizeof(U) * (col_idx + 8), tmp[1]);
                    }
                }
            } else {  // rows 12-15
                for (int j = 0; j < src.width; j++) {
                    U2 tmp[2];
                    tmp[0] = base_types::convertor<U2, T2>::convert(src.tiles[0][j].data[1]);
                    tmp[1] = base_types::convertor<U2, T2>::convert(src.tiles[0][j].data[3]);
                    int col_idx = local_col_idx * 2 + j * 16;
                    if (local_row_idx < GQA_RATIO) {
                        move<U2>::sts(dst_ptr[local_row_idx] + sizeof(U) * col_idx, tmp[0]);
                        move<U2>::sts(dst_ptr[local_row_idx] + sizeof(U) * (col_idx + 8), tmp[1]);
                    }
                }
            }
        }
    }

    template <ducks::rt::row_layout RT>
    __device__ static inline void right_fill(
        RT &dst, const RT &src, const int col_idx,
        const typename base_types::packing<typename RT::dtype>::unpacked_type
            &val = 0) {
        if (col_idx >= dst.cols)
            return;
#pragma unroll
        for (int i = 0; i < dst.height; i++) {
#pragma unroll
            for (int j = 0; j < dst.width; j++) {
#pragma unroll
                for (int k = 0; k < dst.packed_per_tile; k++) {
                    const int col_idx_x = (j * dst.tile_size_col) +
                                          ((k / 2) * 8) +
                                          ((kittens::warp::laneid() % 4) * 2);
                    const int col_idx_y = (j * dst.tile_size_col) +
                                          ((k / 2) * 8) +
                                          ((kittens::warp::laneid() % 4) * 2) + 1;
                    if (col_idx_x >= col_idx) {
                        dst.tiles[i][j].data[k].x = val;
                    } else {
                        dst.tiles[i][j].data[k].x = src.tiles[i][j].data[k].x;
                    }
                    if (col_idx_y >= col_idx) {
                        dst.tiles[i][j].data[k].y = val;
                    } else {
                        dst.tiles[i][j].data[k].y = src.tiles[i][j].data[k].y;
                    }
                }
            }
        }
    }

    // Load Q for GQA_RATIO heads × head_dim elements
    __device__ static inline void
    load_Q_async(q_st &dst, const globals::activations_t &src,
                 const int q_head_start_idx) {
        using T = typename q_st::dtype;
        constexpr int elem_per_memcpy = sizeof(float4) / sizeof(T);  // 8
        constexpr int total_elements = GQA_RATIO * globals::head_dim;
        constexpr int total_memcpys = (total_elements + elem_per_memcpy - 1) / elem_per_memcpy;

        typename globals::activations_t::dtype *src_ptr =
            &src.raw_ptr[q_head_start_idx * globals::head_dim];
        // Always load Q heads at rows 0..GQA_RATIO-1 to avoid
        // overflowing the 16-row tile.
        uint32_t dst_base = static_cast<uint32_t>(__cvta_generic_to_shared(
            &dst.data[0]));

        int laneid = kittens::warp::laneid();

        // Round 1: first 32 copies
        if (laneid < total_memcpys) {
            int flat_elem = laneid * elem_per_memcpy;
            int row = flat_elem / globals::head_dim;
            int col = flat_elem % globals::head_dim;

            asm volatile(
                "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
                ::"r"(dst.idx(dst_base, {row, col})),
                "l"(&src_ptr[row * globals::head_dim + col])
                : "memory");
        }

        // Round 2: remaining copies (elements 256+)
        constexpr int second_round_start = 32;
        if (laneid + second_round_start < total_memcpys) {
            int copy_idx = laneid + second_round_start;
            int flat_elem = copy_idx * elem_per_memcpy;
            int row = flat_elem / globals::head_dim;
            int col = flat_elem % globals::head_dim;

            asm volatile(
                "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
                ::"r"(dst.idx(dst_base, {row, col})),
                "l"(&src_ptr[row * globals::head_dim + col])
                : "memory");
        }

        asm volatile("cp.async.commit_group;\n" ::: "memory");
    }

    struct controller {
        static __device__ int
        release_lid(const globals &g,
                    typename config::instruction_t &instruction, int &query) {
            int ret_order[13] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 1};
            return ret_order[query];
        }
        static __device__ int init_semaphores(const globals &g,
                                              megakernel::state<config> &s) {
            init_semaphore(Q_arrived(s), 0, 1);
            init_semaphore(O_arrived(s), 0, 1);
            init_semaphore(L_arrived(s), 0, 1);
            for (int i = 0; i < NUM_STAGES; i++) {
                init_semaphore(K_arrived(s, i), 0, 1);
                init_semaphore(V_arrived(s, i), 0, 1);
                init_semaphore(K_finished(s, i), 0, 1);
                init_semaphore(V_finished(s, i), 0, 1);
            }
            return 3 + 4 * NUM_STAGES;
        }
    };
    struct loader {
        static __device__ void run(const globals &g, megakernel::state<config> &s) {
            auto laneid = kittens::warp::laneid();
            if (laneid >= 2 && laneid < config::NUM_PAGES) {
                int unused_page = s.pid(laneid);
                s.wait_page_ready(unused_page);
                s.finish_page(unused_page, config::NUM_CONSUMER_WARPS);
            }
        }
    };
    struct launcher {
        static __device__ void wait_for_kv(const globals &g, megakernel::state<config> &s,
                                           parsed_instruction &inst) {
            s.record(megakernel::TEVENT_AT_GMEM_WAIT);

            // head_dim / matvec_block_size = blocks per head
            // Wait for K head blocks
            while (*(volatile int *)&g.Bar[{
                       inst.layer_idx, OPCODE_RMS_QKV_MatVecRopeAppend - 1,
                       globals::num_attention_heads + inst.kv_head_idx}] < globals::head_dim / globals::matvec_block_size) {
                __nanosleep(config::GMEM_SPIN_LOOP_SLEEP_NANOS);
            }

            // Wait for V head blocks
            while (
                *(volatile int *)&g
                     .Bar[{inst.layer_idx, OPCODE_RMS_QKV_MatVecRopeAppend - 1,
                           globals::num_attention_heads +
                               globals::num_kv_heads + inst.kv_head_idx}] < globals::head_dim / globals::matvec_block_size) {
                __nanosleep(config::GMEM_SPIN_LOOP_SLEEP_NANOS);
            }

            s.record(megakernel::TEVENT_DONE_GMEM_WAIT);
        }

        static __device__ void run(const globals &g, megakernel::state<config> &s) {
            if (kittens::warp::laneid() == 0) {
#ifdef KITTENS_BLACKWELL
                s.wait_tensor_ready();
                arrive(s.tensor_finished, config::NUM_CONSUMER_WARPS);
#endif

                parsed_instruction inst{s};
                int seq_len = g.pos_id + 1;
                int total_attn_blocks = (seq_len + globals::kv_block_size - 1) /
                                        globals::kv_block_size;
                int blocks_per_partial =
                    (total_attn_blocks + inst.num_partials - 1) /
                    inst.num_partials;
                int start_blk_idx = inst.partial_idx * blocks_per_partial;
                int end_blk_idx =
                    min(start_blk_idx + blocks_per_partial, total_attn_blocks);

                // Wait for the KV page
                wait_KV_page(s);

                if (start_blk_idx >= end_blk_idx)
                    finish_KV_page(s);

                // Run the pipeline
                for (int i = 0; i + start_blk_idx < end_blk_idx; ++i) {
                    auto cur_blk_idx = start_blk_idx + i;
                    int stage = cur_blk_idx % NUM_STAGES;
                    kv_st &K_smem = get_K_smem(s, stage);
                    kv_st &V_smem = get_V_smem(s, stage);

                    if (i >= NUM_STAGES) {
                        kittens::wait(K_finished(s, stage), (i / NUM_STAGES - 1) % 2);
                        kittens::wait(V_finished(s, stage), (i / NUM_STAGES - 1) % 2);
                    }

                    if (cur_blk_idx == end_blk_idx - 1 &&
                        inst.partial_idx == inst.num_partials - 1) {
                        wait_for_kv(g, s, inst);
                    }

                    kittens::tma::expect(K_arrived(s, stage), K_smem);
                    kittens::tma::load_async<dim::DEPTH, cache_policy::EVICT_FIRST>(
                        K_smem, g.k_cache,
                        {inst.layer_idx, cur_blk_idx, inst.kv_head_idx, 0},
                        K_arrived(s, stage));
                    kittens::tma::expect(V_arrived(s, stage), V_smem);
                    kittens::tma::load_async<dim::DEPTH, cache_policy::EVICT_FIRST>(
                        V_smem, g.v_cache,
                        {inst.layer_idx, cur_blk_idx, inst.kv_head_idx, 0},
                        V_arrived(s, stage));
                }
            }
        }
    };
    struct consumer {
        static __device__ void run(const globals &g, megakernel::state<config> &s) {

            if (kittens::warpid() == 0) {
                parsed_instruction inst{s};
                int q_head_start_idx = inst.kv_head_idx * GQA_RATIO;

                // Wait for Q heads to be ready
                if (kittens::laneid() == 0) {
                    // head_dim / matvec_block_size = blocks per head
                    for (int head_offset = 0; head_offset < GQA_RATIO;
                         head_offset++) {
                        while (*(volatile int *)&g
                                    .Bar[{inst.layer_idx,
                                          OPCODE_RMS_QKV_MatVecRopeAppend - 1,
                                          q_head_start_idx + head_offset}] <
                               globals::head_dim / globals::matvec_block_size) {
                            __nanosleep(config::GMEM_SPIN_LOOP_SLEEP_NANOS);
                        }
                    }
                }
                kittens::warp::sync();

                // Setup
                // Q is always loaded at rows 0..GQA_RATIO-1 of the tile.
                int q_head_local_idx = 0;
                int seq_len = g.pos_id + 1;
                int total_attn_blocks = (seq_len + globals::kv_block_size - 1) /
                                        globals::kv_block_size;
                int blocks_per_partial =
                    (total_attn_blocks + inst.num_partials - 1) /
                    inst.num_partials;
                int start_blk_idx = inst.partial_idx * blocks_per_partial;
                int end_blk_idx =
                    min(start_blk_idx + blocks_per_partial, total_attn_blocks);
                float softmax_temp =
                    g.attn_scale * 1.44269504089f; // 1 / (sqrt(D_h) * ln(2))
                q_rt Q_reg;
                k_rt K_reg;
                v_rt V_reg;
                l_rv L_reg;
                o_rt O_reg;
                attn_fl_rt attn_fl_reg;
                attn_bf_rt attn_bf_reg;
                max_vec_rv max_vec_reg;
                max_vec_rv scaled_max_vec_reg;
                max_vec_rv last_scaled_max_vec_reg;
                max_vec_rv diff_scaled_max_vec_reg;
                norm_vec_rv norm_vec_reg;
                kittens::warp::neg_infty(max_vec_reg);
                kittens::warp::zero(last_scaled_max_vec_reg);
                kittens::warp::zero(norm_vec_reg);
                kittens::warp::zero(O_reg);
                o_sv(&O_smem)[GQA_RATIO] = get_O_smem(s);
                l_sv &L_smem = get_L_smem(s);

                // Load Q
                wait_QOL_page(s);
                q_st &Q_smem = get_Q_smem(s);
                load_Q_async(Q_smem, g.q_post_rope, q_head_start_idx);
                kittens::warp::load_async_wait();
                kittens::warp::load(Q_reg, Q_smem);

                // Run the pipeline
                for (int i = 0; i + start_blk_idx < end_blk_idx; ++i) {
                    int stage = (start_blk_idx + i) % NUM_STAGES;
                    kv_st &K_smem = get_K_smem(s, stage);
                    kv_st &V_smem = get_V_smem(s, stage);

                    // Q @ K.T
                    kittens::warp::zero(attn_fl_reg);
                    kittens::warp::wait(K_arrived(s, stage), (i / NUM_STAGES) % 2);

                    kittens::warp::load(K_reg, K_smem);
                    kittens::warp::mma_ABt(attn_fl_reg, Q_reg, K_reg, attn_fl_reg);
                    kittens::warp::sync();
                    kittens::warp::arrive(K_finished(s, stage));

                    // Mask out invalid positions
                    if ((i + start_blk_idx + 1) * globals::kv_block_size > seq_len)
                        right_fill(attn_fl_reg, attn_fl_reg,
                                   seq_len % globals::kv_block_size,
                                   -999999999999.f);

                    // Row max
                    kittens::warp::row_max(max_vec_reg, attn_fl_reg, max_vec_reg);

                    // Scale
                    kittens::warp::mul(attn_fl_reg, attn_fl_reg, softmax_temp);
                    kittens::warp::mul(scaled_max_vec_reg, max_vec_reg, softmax_temp);

                    // Softmax numerator
                    kittens::warp::sub_row(attn_fl_reg, attn_fl_reg, scaled_max_vec_reg);
                    kittens::warp::exp2(attn_fl_reg, attn_fl_reg);

                    // Softmax denominator update
                    kittens::warp::sub(diff_scaled_max_vec_reg, last_scaled_max_vec_reg,
                              scaled_max_vec_reg);
                    kittens::warp::exp2(diff_scaled_max_vec_reg, diff_scaled_max_vec_reg);

                    // Accumulate A @ V
                    kittens::warp::mul_row(O_reg, O_reg, diff_scaled_max_vec_reg);
                    kittens::warp::wait(V_arrived(s, stage), (i / NUM_STAGES) % 2);

                    kittens::warp::load(V_reg, V_smem);
                    kittens::warp::copy(attn_bf_reg, attn_fl_reg);
                    kittens::warp::mma_AB(O_reg, attn_bf_reg, V_reg, O_reg);
                    kittens::warp::sync();
                    kittens::warp::arrive(V_finished(s, stage));

                    // Denominator accumulate
                    kittens::warp::mul(norm_vec_reg, norm_vec_reg, diff_scaled_max_vec_reg);
                    kittens::warp::row_sum(norm_vec_reg, attn_fl_reg, norm_vec_reg);

                    kittens::warp::copy(last_scaled_max_vec_reg, scaled_max_vec_reg);
                }

                // Finish
                kittens::warp::sync();

                if (start_blk_idx < end_blk_idx) {
                    finish_KV_page(s);
                    kittens::warp::div_row(O_reg, O_reg, norm_vec_reg);
                    kittens::warp::log2(L_reg, norm_vec_reg);
                    kittens::warp::add(L_reg, L_reg, last_scaled_max_vec_reg);
                } else {
                    kittens::warp::neg_infty(L_reg);
                }

                // Store: GQA_RATIO heads
                store_gqa_rows(O_smem, O_reg, q_head_local_idx);
                kittens::warp::sync();

                kittens::warp::arrive(O_arrived(s));
                kittens::warp::store(L_smem, L_reg);
                kittens::warp::sync();
                kittens::warp::arrive(L_arrived(s));
            }
        }
    };
    struct storer {

        static inline __device__ void
        store_o_skip(const globals &g, megakernel::state<config> &s, int q_head_start_idx) {
            auto O_smem = get_O_smem(s);

            if (kittens::laneid() == 0) {
                kittens::wait(O_arrived(s), 0);
                s.record(megakernel::TEVENT_OUTPUT_READY);
            }
            kittens::warp::sync();

            kittens::rv_bf<globals::head_dim> O_bf;
            for (int head_offset = 0; head_offset < GQA_RATIO; head_offset++) {
                auto &smem_fl = O_smem[head_offset];
                auto &smem_bf = *reinterpret_cast<o_sv_bf *>(&smem_fl);

                kittens::warp::load(O_bf, smem_fl);
                kittens::warp::sync();
                kittens::warp::store(smem_bf, O_bf);
                kittens::warp::sync();
            }

            if (kittens::laneid() == 0) {
                for (int head_offset = 0; head_offset < GQA_RATIO; head_offset++) {
                    auto &smem_bf = *reinterpret_cast<o_sv_bf *>(&O_smem[head_offset]);
                    kittens::tma::store_async<cache_policy::EVICT_LAST>(
                        g.attn_out, smem_bf, {q_head_start_idx + head_offset});
                }
            }
        }

        static inline __device__ void
        store_o_no_skip(const globals &g, megakernel::state<config> &s,
                        int q_head_start_idx, parsed_instruction &inst) {
            if (kittens::laneid() == 0) {
                o_sv(&O_smem)[GQA_RATIO] = get_O_smem(s);
                kittens::wait(O_arrived(s), 0);
                s.record(megakernel::TEVENT_OUTPUT_READY);

                for (int head_offset = 0; head_offset < GQA_RATIO; head_offset++) {
                    kittens::tma::store_async<cache_policy::EVICT_LAST>(
                        g.attn_out_intermediates, O_smem[head_offset],
                        {0, q_head_start_idx + head_offset, inst.partial_idx, 0});
                }
            }
        }

        static __device__ void run(const globals &g, megakernel::state<config> &s) {
            parsed_instruction inst{s};
            int laneid = kittens::warp::laneid();
            int q_head_start_idx = inst.kv_head_idx * GQA_RATIO;
            // Q is always loaded at rows 0..GQA_RATIO-1.
            int q_head_vec_start_idx = 0;

            auto skip_attn_reduction = g.skip_attn_reduction;

            if (skip_attn_reduction) {
                store_o_skip(g, s, q_head_start_idx);
            } else {
                store_o_no_skip(g, s, q_head_start_idx, inst);
            }

            // Store LSE
            if (laneid < GQA_RATIO && !skip_attn_reduction) {
                l_sv &L_smem = get_L_smem(s);
                kittens::wait(L_arrived(s), 0);

                float tmp;
                uint32_t src_ptr =
                    static_cast<uint32_t>(__cvta_generic_to_shared(
                        &L_smem.data[q_head_vec_start_idx + laneid]));
                float *dst_ptr =
                    (float *)&g.attn_lse_intermediates
                        .raw_ptr[(q_head_start_idx + laneid) *
                                     g.attn_lse_intermediates.cols() +
                                 inst.partial_idx];
                asm volatile("ld.shared.f32 %0, [%1];\n"
                             : "=f"(tmp)
                             : "r"(src_ptr));
                asm volatile("st.global.f32 [%0], %1;\n"
                             :
                             : "l"(dst_ptr), "f"(tmp));
            }
            kittens::warp::sync();
            asm volatile("fence.acq_rel.gpu;");

            kittens::tma::store_async_wait();
            if (laneid == 0) {
                s.record(123 + laneid);
                finish_QOL_page(s);
            }

            if (laneid == 0) {
                s.record(megakernel::TEVENT_AT_GMEM_STORE);

                if (skip_attn_reduction) {
                    atomicAdd(&g.Bar[{inst.layer_idx,
                                      OPCODE_AttentionReduction - 1, 0}],
                              GQA_RATIO);
                } else {
                    atomicAdd(&g.Bar[{inst.layer_idx, opcode - 1,
                                      inst.kv_head_idx}],
                              1);
                }

                s.record(megakernel::TEVENT_DONE_GMEM_STORE);
            }
        }
    };
};
