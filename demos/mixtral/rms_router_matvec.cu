#pragma once

#include "mixtral.cuh"
#include "utils.cuh"
#include "matvec_pipeline.cuh"

using namespace kittens;
using namespace megakernel;

// rms_router_upgate (opcode 5):
//   Fused: RMS norm + router top-2 + gate+up matvec + SiLU for both selected experts.
//
//   Phase 1 (consumer pre-pipeline): RMS norm + router selection
//     1. Wait for all o_proj blocks (Bar[layer, OPCODE_OProj-1, 0] >= expected)
//     2. RMS norm hidden_states with ffn_ln_weights -> activations_vec
//     3. Store normed activations to router_normed_hidden
//     4. Warp 0 computes 8 dot products -> softmax -> top-2 -> renormalize
//     5. Write selected_expert_indices and selected_expert_scores
//
//   Phase 2 (pipeline): gate+up matvec + SiLU for both selected experts
//     iters = 2 * num_blocks * num_experts_per_tok (gate+up interleaved x 2 experts)
//     load_iter selects expert index at runtime from selected_expert_indices
//     store: on odd iters, reduces up+gate, applies SiLU, stores to expert_silu_out
//
//   Barrier: signals Bar[layer, OPCODE_RmsRouterUpgate-1, 0] += num_blocks

template <typename Config, typename Globals> struct rms_router_upgate {
    static constexpr int opcode = OPCODE_RmsRouterUpgate;
    static constexpr int EXPECTED_OPROJ_ARRIVALS = Globals::hidden_dim / Globals::matvec_block_size;

    // Expert routing info stored in scratch memory (after output pipeline scratch).
    // This avoids a global memory round-trip for expert indices within this SM.
    static constexpr int EXPERT_SCRATCH_OFFSET =
        3 /* OUTPUT_PIPELINE_STAGES */ * 16 * sizeof(float) * Config::NUM_CONSUMER_WARPS;
    struct expert_routing_info {
        int indices[Globals::num_experts_per_tok];
    };
    __device__ static inline expert_routing_info *
    get_routing_info(megakernel::state<Config> &s) {
        return reinterpret_cast<expert_routing_info *>(
            (uint8_t *)s.scratch() + EXPERT_SCRATCH_OFFSET);
    }

    // Pipeline uses half of hidden_dim for REDUCTION_DIM to get 3-stage
    // pipelining (STAGE_PAGES=4, INPUT_PIPELINE_STAGES=3) instead of 1-stage
    // (STAGE_PAGES=8) when reducing over the full hidden_dim=4096.
    static constexpr int REDUCTION_DIM = Globals::hidden_dim / 2;
    static constexpr int INNER_TILES = Globals::hidden_dim / REDUCTION_DIM;

    struct parsed_instruction {
        int layer_idx, start_block_idx, num_blocks;
        // iters: total loader pipeline iterations (used by release_lid).
        // iters == iters_outer * INNER_TILES.
        int iters;
        int iters_outer, loader_iters;
        __device__ inline parsed_instruction(
            typename Config::instruction_t &instr) {
            layer_idx       = instr[1];
            start_block_idx = instr[2];
            num_blocks      = instr[3];
            // gate + up interleaved x 2 experts
            iters           = 2 * num_blocks * Globals::num_experts_per_tok
                              * INNER_TILES;
            iters_outer     = 2 * num_blocks * Globals::num_experts_per_tok;
            loader_iters    = iters;
        }
        __device__ inline parsed_instruction(megakernel::state<Config> &s)
            : parsed_instruction(s.instruction()) {}
    };

    struct pipeline_specifics {
        static __device__ inline void gmem_wait(const Globals &g,
                                                megakernel::state<Config> &s) {}

        template <int TW>
        static __device__ inline void
        load_iter(megakernel::state<Config> &s, const Globals &g,
                  parsed_instruction &inst, int iter, int col_idx,
                  kittens::st_bf<16, TW> &weight_chunk,
                  kittens::semaphore &sem) {
            // With inner tiling, iter is a flat loader iteration index:
            //   outer_iter  = iter / INNER_TILES  (which up/gate x block x expert)
            //   tile_batch  = iter % INNER_TILES  (which column half)
            // outer_iter layout: for each expert_loop in [0, num_experts_per_tok):
            //   for each block in [0, num_blocks):
            //     even sub-iter: up weights
            //     odd sub-iter: gate weights
            int outer_iter = iter / INNER_TILES;
            int tile_batch = iter % INNER_TILES;

            int expert_loop = outer_iter / (2 * inst.num_blocks);
            int within_expert = outer_iter % (2 * inst.num_blocks);
            int block_offset = within_expert / 2;
            int block_idx = inst.start_block_idx + block_offset;

            // Get selected expert index from scratch (written by consumer on this SM)
            int expert_idx = get_routing_info(s)->indices[expert_loop];
            int depth = inst.layer_idx * Globals::num_experts + expert_idx;

            // col_coord: tile_batch selects which REDUCTION_DIM-wide half,
            // col_idx selects the page within that half.
            int col_coord = tile_batch * pipeline::STAGE_PAGES + col_idx;

            if (within_expert % 2 == 0) {
                // even: up weights
                kittens::tma::load_async<dim::ROW, cache_policy::EVICT_FIRST>(
                    weight_chunk, g.expert_up_weights,
                    {0, depth, block_idx, col_coord}, sem);
            } else {
                // odd: gate weights
                kittens::tma::load_async<dim::ROW, cache_policy::EVICT_FIRST>(
                    weight_chunk, g.expert_gate_weights,
                    {0, depth, block_idx, col_coord}, sem);
            }
        }

        template <int TW>
        static __device__ inline void
        prefetch_iter(megakernel::state<Config> &s, const Globals &g,
                      parsed_instruction &inst, int iter, int col_idx,
                      kittens::st_bf<16, TW> &weight_chunk) {
            int outer_iter = iter / INNER_TILES;
            int tile_batch = iter % INNER_TILES;

            int expert_loop = outer_iter / (2 * inst.num_blocks);
            int within_expert = outer_iter % (2 * inst.num_blocks);
            int block_offset = within_expert / 2;
            int block_idx = inst.start_block_idx + block_offset;

            int expert_idx = get_routing_info(s)->indices[expert_loop];
            int depth = inst.layer_idx * Globals::num_experts + expert_idx;

            int col_coord = tile_batch * pipeline::STAGE_PAGES + col_idx;

            if (within_expert % 2 == 0) {
                kittens::tma::prefetch<dim::ROW, cache_policy::EVICT_FIRST>(
                    weight_chunk, g.expert_up_weights,
                    {0, depth, block_idx, col_coord});
            } else {
                kittens::tma::prefetch<dim::ROW, cache_policy::EVICT_FIRST>(
                    weight_chunk, g.expert_gate_weights,
                    {0, depth, block_idx, col_coord});
            }
        }

        static __device__ inline void store(megakernel::state<Config> &s,
                                            const Globals &g,
                                            parsed_instruction &inst,
                                            int output_idx, int output_stage) {
            // Even iter = up weights pass, no output yet
            if (output_idx % 2 == 0) {
                return;
            }

            int expert_loop = output_idx / (2 * inst.num_blocks);
            int within_expert = output_idx % (2 * inst.num_blocks);
            int true_output_idx = within_expert / 2;
            auto prev_output_idx   = output_idx - 1;
            auto prev_output_stage = prev_output_idx % pipeline::OUTPUT_PIPELINE_STAGES;

            int block_idx = inst.start_block_idx + true_output_idx;
            int expert_idx = get_routing_info(s)->indices[expert_loop];

            uint8_t *output_scratch_start =
                pipeline::get_output_start(s, output_stage);
            uint8_t *prev_output_scratch_start =
                pipeline::get_output_start(s, prev_output_stage);

            kittens::sv_bf<16> &out_smem =
                *reinterpret_cast<kittens::sv_bf<16> *>(output_scratch_start);

            kittens::rv_fl<16> up_out, gate_out, gate_scratch;

            matvec_reduce<Config, kittens::sv_fl<16>, kittens::rv_fl<16>,
                          pipeline::SCRATCH_BYTES_PER_WARP>(
                prev_output_scratch_start, up_out);
            matvec_reduce<Config, kittens::sv_fl<16>, kittens::rv_fl<16>,
                          pipeline::SCRATCH_BYTES_PER_WARP>(
                output_scratch_start, gate_out);

            // SiLU(gate)
            kittens::warp::mul(gate_scratch, gate_out, -1.f);
            kittens::warp::exp(gate_scratch, gate_scratch);
            kittens::warp::add(gate_scratch, gate_scratch, 1.f);
            kittens::warp::div(gate_out, gate_out, gate_scratch);

            // gating: SiLU(gate) * up
            kittens::warp::mul(gate_out, up_out, gate_out);

            kittens::warp::sync();
            kittens::warp::store(out_smem, gate_out);
            kittens::warp::sync();

            if (kittens::laneid() == 0) {
                s.record(megakernel::TEVENT_OUTPUT_READY);
                kittens::tma::store_async<cache_policy::EVICT_LAST>(
                    g.expert_silu_out, out_smem,
                    {0, 0, expert_idx, block_idx});
                kittens::tma::store_async_read_wait();
            }

            kittens::warp::sync();
        }
    };

    using pipeline = matvec_pipeline<Config, Globals, parsed_instruction,
                                     pipeline_specifics, REDUCTION_DIM>;

    // Semaphore for RMS norm weights loaded by loader
    __device__ static inline kittens::semaphore &
    rms_weights_arrived(megakernel::state<Config> &s) {
        return s.semaphores()[pipeline::SEM_COUNT];
    }

    // Semaphore signaled by consumer after writing selected_expert_indices/scores.
    // The loader must wait on this before entering the pipeline loop (which reads
    // selected_expert_indices to determine which expert's weights to load).
    __device__ static inline kittens::semaphore &
    router_done(megakernel::state<Config> &s) {
        return s.semaphores()[pipeline::SEM_COUNT + 1];
    }

    static constexpr int SEM_COUNT = pipeline::SEM_COUNT + 2;

    struct controller {
        static __device__ int
        release_lid(const Globals &g,
                    typename Config::instruction_t &instruction, int &query) {
            return pipeline::release_lid(g, instruction, query);
        }
        static __device__ int init_semaphores(const Globals &g,
                                              megakernel::state<Config> &s) {
            pipeline::init_semaphores(s);
            init_semaphore(rms_weights_arrived(s), 1);
            // router_done: signaled by 1 consumer warp (warp 0), waited by loader (1 warp)
            init_semaphore(router_done(s), 1);
            return SEM_COUNT;
        }
    };

    struct loader {
        static __device__ void run(const Globals &g,
                                   megakernel::state<Config> &s) {
            parsed_instruction inst{s};

            // First, load RMS norm weights into the activation page
            if (kittens::laneid() == 0) {
                int activation_page = pipeline::get_activation_page(s);
                s.wait_page_ready(activation_page);

                auto &rms_smem = *reinterpret_cast<kittens::sv_bf<Globals::hidden_dim> *>(
                    s.pages[activation_page].ptr(sizeof(kittens::sv_bf<Globals::hidden_dim>)));

                auto &sem = rms_weights_arrived(s);
                kittens::tma::expect(sem, rms_smem);
                kittens::tma::load_async<cache_policy::EVICT_LAST>(
                    rms_smem, g.ffn_ln_weights, {inst.layer_idx, 0}, sem);

                // Read speculative expert prediction (NO spin wait — just load
                // from gmem; the prediction instruction ran during attention)
                int pred_idx0 = g.predicted_expert_indices.raw_ptr[0];
                int pred_idx1 = g.predicted_expert_indices.raw_ptr[1];

                // Prefetch up/gate weight tiles for all blocks and inner tiles
                // for each predicted expert into L2 cache.
                auto &weight_template =
                    reinterpret_cast<kittens::st_bf<16, pipeline::TILE_WIDTH> &>(
                        s.pages[activation_page]);

                for (int e = 0; e < Globals::num_experts_per_tok; e++) {
                    int expert_idx = (e == 0) ? pred_idx0 : pred_idx1;
                    int depth = inst.layer_idx * Globals::num_experts + expert_idx;
                    for (int b = 0; b < inst.num_blocks; b++) {
                        int block_idx = inst.start_block_idx + b;
                        for (int tile_batch = 0; tile_batch < INNER_TILES; tile_batch++) {
                            for (int p = 0; p < pipeline::STAGE_PAGES; p++) {
                                int col_coord = tile_batch * pipeline::STAGE_PAGES + p;
                                kittens::tma::prefetch<dim::ROW, cache_policy::EVICT_FIRST>(
                                    weight_template, g.expert_up_weights,
                                    {0, depth, block_idx, col_coord});
                            }
                            for (int p = 0; p < pipeline::STAGE_PAGES; p++) {
                                int col_coord = tile_batch * pipeline::STAGE_PAGES + p;
                                kittens::tma::prefetch<dim::ROW, cache_policy::EVICT_FIRST>(
                                    weight_template, g.expert_gate_weights,
                                    {0, depth, block_idx, col_coord});
                            }
                        }
                    }
                }

                // Wait for the consumer to finish computing the router and
                // writing selected_expert_indices/scores. The pipeline's
                // load_iter reads these indices to determine which expert's
                // weights to load, so we must not start loading until they
                // are written.
                kittens::wait(router_done(s), 0);
            }

            // Custom loader with inner-tile iteration.
            // Total loader iterations = iters_outer * INNER_TILES.
            // Each iteration loads STAGE_PAGES pages covering REDUCTION_DIM columns.
            auto needed_pages =
                1 + min(inst.loader_iters, pipeline::INPUT_PIPELINE_STAGES) *
                    pipeline::STAGE_PAGES;

            if (kittens::laneid() == 0) {
                int input_stage = 0;

                for (int iter = 0; iter < inst.loader_iters; iter++) {
                    kittens::wait(
                        pipeline::weights_finished(s, input_stage),
                        (iter % (2 * pipeline::INPUT_PIPELINE_STAGES)) <
                            pipeline::INPUT_PIPELINE_STAGES);

                    auto &sem = pipeline::weights_arrived(s, input_stage);
                    kittens::tma::expect_bytes(
                        sem,
                        sizeof(kittens::bf16) * REDUCTION_DIM * 16);

#pragma unroll
                    for (int i = 0; i < pipeline::STAGE_PAGES; i++) {
                        int weight_page =
                            pipeline::get_weight_page(s, input_stage, i);
                        if (iter < pipeline::INPUT_PIPELINE_STAGES) {
                            s.wait_page_ready(weight_page);
                        }
                        auto &weight_chunk =
                            reinterpret_cast<kittens::st_bf<16, pipeline::TILE_WIDTH> &>(
                                s.pages[weight_page]);

                        if (iter == 0 && i == 0) {
                            s.record(megakernel::TEVENT_FIRST_LOAD);
                        } else if (iter == inst.loader_iters - 1 &&
                                   i == pipeline::STAGE_PAGES - 1) {
                            s.record(megakernel::TEVENT_LAST_LOAD);
                        }

                        pipeline_specifics::load_iter(
                            s, g, inst, iter, i, weight_chunk, sem);
                    }

                    // Prefetch next iteration's weights into L2
                    {
                        int prefetch_iter = iter + 1;
                        if (prefetch_iter < inst.loader_iters) {
#pragma unroll
                            for (int i = 0; i < pipeline::STAGE_PAGES; i++) {
                                auto &weight_chunk =
                                    reinterpret_cast<kittens::st_bf<16, pipeline::TILE_WIDTH> &>(
                                        s.pages[pipeline::get_weight_page(s, input_stage, i)]);
                                pipeline_specifics::prefetch_iter(
                                    s, g, inst, prefetch_iter, i, weight_chunk);
                            }
                        }
                    }

                    input_stage = (input_stage + 1) % pipeline::INPUT_PIPELINE_STAGES;
                }
            } else if (kittens::laneid() >= needed_pages &&
                       kittens::laneid() < Config::NUM_PAGES) {
                auto pid = s.pid(kittens::laneid());
                s.wait_page_ready(pid);
                s.finish_page(pid, Config::NUM_CONSUMER_WARPS);
            }
        }
    };

    struct launcher {
        static __device__ void run(const Globals &g,
                                   megakernel::state<Config> &s) {
            if (kittens::laneid() == 0) {
#ifdef KITTENS_BLACKWELL
                s.wait_tensor_ready();
                arrive(s.tensor_finished, Config::NUM_CONSUMER_WARPS);
#endif
            }
        }
    };

    struct consumer {
        static __device__ void run(const Globals &g,
                                   megakernel::state<Config> &s) {
            using sv_t = kittens::sv_bf<pipeline::REDUCTION_DIM_PER_WARP>;
            using rv_t = kittens::rv_fl<pipeline::REDUCTION_DIM_PER_WARP>;
            parsed_instruction inst{s};

            // ---- Phase 1: RMS norm + Router ----

            // Warp 0, lane 0 waits for o_proj barrier
            if (kittens::warpid() == 0 && kittens::laneid() == 0) {
                int activation_page = pipeline::get_activation_page(s);
                s.wait_page_ready(activation_page);

                s.record(megakernel::TEVENT_AT_GMEM_WAIT);
                while (*(volatile int *)&g.Bar[{inst.layer_idx, OPCODE_OProj - 1, 0}] <
                       EXPECTED_OPROJ_ARRIVALS) {
                    __nanosleep(Config::GMEM_SPIN_LOOP_SLEEP_NANOS);
                }
                s.record(megakernel::TEVENT_DONE_GMEM_WAIT);
            }
            kittens::group<Config::NUM_CONSUMER_WARPS>::sync(3);

            // Wait for RMS norm weights
            kittens::wait(rms_weights_arrived(s), 0);

            int activation_page = pipeline::get_activation_page(s);

            // Full-size path: RMS norm using SMEM pipeline
            constexpr int CHUNK = Globals::hidden_dim / Config::NUM_CONSUMER_WARPS;
            using sv_chunk = kittens::sv_bf<CHUNK>;
            using rv_chunk = kittens::rv_fl<CHUNK>;

            sv_chunk &act_smem = reinterpret_cast<sv_chunk *>(
                s.pages[activation_page].ptr())[kittens::warpid()];
            sv_chunk &nw_smem = reinterpret_cast<sv_chunk *>(
                s.pages[activation_page].ptr(sizeof(kittens::sv_bf<Globals::hidden_dim>)))[kittens::warpid()];

            kittens::warp::load(act_smem, g.hidden_states,
                                kittens::coord<>{kittens::warpid() * CHUNK});
            kittens::warp::sync();

            rv_chunk activations_vec = rms_norm<Config, Globals>(
                nw_smem, act_smem, g.rms_norm_eps, s.scratch());
            kittens::warp::sync();

            // Store normed chunk to router_normed_hidden
            {
                sv_chunk &tmp_smem = act_smem;
                kittens::warp::store(tmp_smem, activations_vec);
                kittens::warp::sync();
                kittens::warp::store(g.router_normed_hidden, tmp_smem,
                                     kittens::coord<>{kittens::warpid() * CHUNK});
            }

            kittens::group<Config::NUM_CONSUMER_WARPS>::sync(1);

            // All warps compute router dot product partials over their chunk
            {
                float partial_dots[Globals::num_experts];
                int lane = kittens::laneid();
                int chunk_start = kittens::warpid() * CHUNK;

                for (int e = 0; e < Globals::num_experts; e++) {
                    float dot = 0.f;
                    int row_base = (inst.layer_idx * Globals::num_experts + e) *
                                   Globals::hidden_dim;
                    for (int i = lane; i < CHUNK; i += 32) {
                        float nv = float(g.router_normed_hidden.raw_ptr[chunk_start + i]);
                        float wv = float(g.router_weights.raw_ptr[row_base + chunk_start + i]);
                        dot += nv * wv;
                    }
                    for (int mask = 16; mask >= 1; mask >>= 1)
                        dot += __shfl_xor_sync(0xffffffff, dot, mask);
                    partial_dots[e] = dot;
                }

                // Write partials to scratch: layout [expert][warpid]
                float *scratch = (float *)s.scratch();
                if (kittens::laneid() == 0) {
                    for (int e = 0; e < Globals::num_experts; e++) {
                        scratch[e * Config::NUM_CONSUMER_WARPS + kittens::warpid()] = partial_dots[e];
                    }
                }
                kittens::group<Config::NUM_CONSUMER_WARPS>::sync(2);

                // Warp 0, lane 0 reduces partials and does softmax + top-2
                if (kittens::warpid() == 0 && kittens::laneid() == 0) {
                    float logits[Globals::num_experts];
                    for (int e = 0; e < Globals::num_experts; e++) {
                        float sum = 0.f;
                        for (int w = 0; w < Config::NUM_CONSUMER_WARPS; w++) {
                            sum += scratch[e * Config::NUM_CONSUMER_WARPS + w];
                        }
                        logits[e] = sum;
                    }

                    float max_val = logits[0];
                    for (int e = 1; e < Globals::num_experts; e++)
                        max_val = fmaxf(max_val, logits[e]);
                    float sum_exp = 0.f;
                    float probs[Globals::num_experts];
                    for (int e = 0; e < Globals::num_experts; e++) {
                        probs[e] = __expf(logits[e] - max_val);
                        sum_exp += probs[e];
                    }
                    for (int e = 0; e < Globals::num_experts; e++)
                        probs[e] /= sum_exp;

                    int idx0 = 0, idx1 = 1;
                    if (probs[idx1] > probs[idx0]) { int tmp = idx0; idx0 = idx1; idx1 = tmp; }
                    for (int e = 2; e < Globals::num_experts; e++) {
                        if (probs[e] > probs[idx0]) { idx1 = idx0; idx0 = e; }
                        else if (probs[e] > probs[idx1]) { idx1 = e; }
                    }

                    float top2_sum = probs[idx0] + probs[idx1];
                    // Write to scratch for loader/storer on this SM
                    auto *routing = get_routing_info(s);
                    routing->indices[0] = idx0;
                    routing->indices[1] = idx1;
                    // Write to gmem for cross-SM consumers (expert_downproj_fused)
                    g.selected_expert_indices.raw_ptr[0] = idx0;
                    g.selected_expert_indices.raw_ptr[1] = idx1;
                    g.selected_expert_scores.raw_ptr[0] = __float2bfloat16(probs[idx0] / top2_sum);
                    g.selected_expert_scores.raw_ptr[1] = __float2bfloat16(probs[idx1] / top2_sum);
                }
            }

            // All warps sync + fence so all warps see router outputs
            kittens::group<Config::NUM_CONSUMER_WARPS>::sync(4);

            // Signal the loader that expert indices are ready
            if (kittens::warpid() == 0 && kittens::laneid() == 0) {
                arrive(router_done(s));
            }

            // Release activation page — the full path reloads activations
            // per inner tile from router_normed_hidden via the activation page.
            s.warp_finish_page(activation_page, 1);

            // ---- Phase 2: Inner-tile UpGate pipeline ----
            // Outer loop: iters_outer = 2 * num_blocks * num_experts_per_tok
            //   (up/gate interleaved, same as before)
            // Inner loop: INNER_TILES tile-batches of REDUCTION_DIM columns.
            //   Each inner tile loads REDUCTION_DIM columns of weights and
            //   the corresponding activation slice, does matvec, accumulates.
            // After inner loop: writes accumulated result to output scratch.
            // Storer applies SiLU fusion on odd outer iters (unchanged).
            constexpr int RDPW = pipeline::REDUCTION_DIM_PER_WARP;
            constexpr int WARPS_PER_PAGE =
                Config::NUM_CONSUMER_WARPS / pipeline::STAGE_PAGES;
            const int page_index = kittens::warpid() / WARPS_PER_PAGE;

            int input_stage = 0, output_stage = 0;

            for (int outer = 0; outer < inst.iters_outer; outer++) {
                kittens::rv_fl<16> warp_acc;
                kittens::warp::zero(warp_acc);

                kittens::sv_fl<16> &warp_scratch =
                    *reinterpret_cast<kittens::sv_fl<16> *>(
                        pipeline::get_output_start(s, output_stage) +
                        (kittens::warpid() * pipeline::SCRATCH_BYTES_PER_WARP));

                for (int t = 0; t < INNER_TILES; t++) {
                    int iter = outer * INNER_TILES + t;

                    // Wait for weight pages
                    kittens::wait(
                        pipeline::weights_arrived(s, input_stage),
                        (iter % (2 * pipeline::INPUT_PIPELINE_STAGES)) >=
                            pipeline::INPUT_PIPELINE_STAGES);

                    // Wait for output scratch free (first inner tile only)
                    if (t == 0) {
                        kittens::wait(
                            pipeline::outputs_finished(s, output_stage),
                            (outer % (2 * pipeline::OUTPUT_PIPELINE_STAGES)) <
                                pipeline::OUTPUT_PIPELINE_STAGES);
                    }

                    // Get weight subtile for this warp
                    int weight_page =
                        pipeline::get_weight_page(s, input_stage, page_index);
                    kittens::st_bf<16, RDPW> &weights =
                        reinterpret_cast<kittens::st_bf<16, RDPW> *>(
                            s.pages[weight_page].ptr())
                            [kittens::warpid() % WARPS_PER_PAGE];

                    // Load activation slice for this tile-batch from gmem
                    // via the activation page (shared memory).
                    int act_col = t * REDUCTION_DIM + kittens::warpid() * RDPW;
                    kittens::rv_fl<RDPW> act_vec;
                    {
                        int act_page = pipeline::get_activation_page(s);
                        kittens::sv_bf<RDPW> &act_smem =
                            reinterpret_cast<kittens::sv_bf<RDPW> *>(
                                s.pages[act_page].ptr())[kittens::warpid()];
                        kittens::warp::load(
                            act_smem, g.router_normed_hidden,
                            kittens::coord<>{act_col});
                        kittens::warp::sync();
                        kittens::warp::load(act_vec, act_smem);
                        kittens::warp::sync();
                    }

                    if (outer == 0 && t == 0) {
                        s.record(megakernel::TEVENT_FIRST_USE);
                    }

                    // Matvec and accumulate
                    matvec(warp_scratch, weights, act_vec);

                    kittens::rv_fl<16> tile_partial;
                    kittens::warp::load(tile_partial, warp_scratch);
                    kittens::warp::add(warp_acc, warp_acc, tile_partial);

                    // Signal that we are done with this weight stage
                    kittens::warp::arrive(
                        pipeline::weights_finished(s, input_stage));

                    // Release weight pages in drain phase
                    if (iter >= inst.loader_iters -
                                    pipeline::INPUT_PIPELINE_STAGES) {
#pragma unroll
                        for (int j = 0; j < pipeline::STAGE_PAGES; j++) {
                            s.warp_finish_page(
                                pipeline::get_weight_page(s, input_stage, j),
                                1);
                        }
                    }

                    input_stage =
                        (input_stage + 1) % pipeline::INPUT_PIPELINE_STAGES;
                }  // end inner tile loop

                if (outer == inst.iters_outer - 1) {
                    s.record(megakernel::TEVENT_LAST_USE);
                }

                // Write accumulated result to scratch for the storer.
                kittens::warp::store(warp_scratch, warp_acc);
                kittens::warp::sync();

                // All warps signal that outputs have arrived for this block.
                kittens::warp::arrive(
                    pipeline::outputs_arrived(s, output_stage));

                output_stage =
                    (output_stage + 1) % pipeline::OUTPUT_PIPELINE_STAGES;
            }  // end outer loop
        }
    };

    struct storer {
        static __device__ void run(const Globals &g,
                                   megakernel::state<Config> &s) {
            // Custom storer iterating over iters_outer.
            // The store function is called with outer iteration index;
            // SiLU fusion on odd iters remains unchanged.
            parsed_instruction inst{s};
            int output_stage = 0;

            if (kittens::laneid() == 0) {
                s.record(megakernel::TEVENT_AT_GMEM_STORE);
            }

            for (int outer = 0; outer < inst.iters_outer; outer++) {
                auto &sem = pipeline::outputs_arrived(s, output_stage);
                bool bit = (outer % (2 * pipeline::OUTPUT_PIPELINE_STAGES)) >=
                           pipeline::OUTPUT_PIPELINE_STAGES;
                kittens::wait(sem, bit);

                if (outer == 0) {
                    s.record(megakernel::TEVENT_FIRST_STORE);
                } else if (outer == inst.iters_outer - 1) {
                    s.record(megakernel::TEVENT_LAST_STORE);
                }

                pipeline_specifics::store(s, g, inst, outer, output_stage);

                kittens::warp::arrive(
                    pipeline::outputs_finished(s, output_stage));

                output_stage =
                    (output_stage + 1) % pipeline::OUTPUT_PIPELINE_STAGES;
            }

            kittens::warp::sync();
            if (kittens::laneid() == 0) {
                kittens::tma::store_async_wait();
                atomicAdd(&g.Bar[{inst.layer_idx, opcode - 1, 0}],
                          inst.num_blocks);
                s.record(megakernel::TEVENT_DONE_GMEM_STORE);
            }
        }
    };
};
