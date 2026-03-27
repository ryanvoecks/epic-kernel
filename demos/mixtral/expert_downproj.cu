#pragma once

#include "mixtral.cuh"
#include "utils.cuh"
#include "matvec_pipeline.cuh"

using namespace kittens;
using namespace megakernel;

// expert_downproj_fused (opcode 6):
//   Fused: downproj for both selected experts in a single instruction.
//   Computes: hidden_states += score * (down_weights @ expert_silu_out)
//   for each of the 2 selected experts.
//
//   Parsed instruction: [opcode=6, layer_idx, start_block_idx, end_block_idx,
//                        reduction_block_idx=0]
//
//   SMALL_TEST build (intermediate_dim=1024):
//     Uses matvec_pipeline<REDUCTION_DIM=1024> unchanged.
//     STAGE_PAGES=2, TILE_WIDTH=512, IPS=6, RDPW=64.
//     iters = blocks_per_expert * num_experts_per_tok (one weight page per iter).
//
//   Full build (intermediate_dim=14336):
//     Streams 14 tile-batches of 1024 columns through the weight pipeline.
//     Loader iters = blocks_per_expert * num_experts_per_tok * INNER_TILES = 7168.
//     Consumer outer loop = blocks_per_expert * num_experts_per_tok = 512.
//     For each outer iter the consumer accumulates float32 partials across
//     INNER_TILES=14 tile-batches, then writes to scratch and signals outputs_arrived.
//     Storer fires once per outer iter (512 times).
//     Activations are loaded from global memory per tile-batch (Option A).

template <typename Config, typename Globals> struct expert_downproj_fused {
    static constexpr int opcode      = OPCODE_ExpertDownProjFused;
    static constexpr int prev_opcode = OPCODE_RmsRouterUpgate;

    // RmsRouterUpgate signals intermediate_dim/matvec_block_size arrivals per SM.
    static constexpr int EXPECTED_ARRIVAL_COUNT =
        Globals::intermediate_dim / Globals::matvec_block_size;

    // -----------------------------------------------------------------------
    // Shared constants
    // -----------------------------------------------------------------------
    static constexpr int TILE_WIDTH    = 512;  // columns per page (= PAGE_SIZE / (16*2))
    static constexpr int STAGE_PAGES   = 2;    // pages per pipeline stage (= 1024 cols)
    static constexpr int STAGE_COLS    = TILE_WIDTH * STAGE_PAGES;  // 1024
    static constexpr int WARPS_PER_PAGE = Config::NUM_CONSUMER_WARPS / STAGE_PAGES; // 8
    static constexpr int RDPW          = STAGE_COLS / Config::NUM_CONSUMER_WARPS;   // 64
    static constexpr int INPUT_PIPELINE_STAGES = (Config::NUM_PAGES - 1) / STAGE_PAGES; // 6
    static constexpr int OUTPUT_PIPELINE_STAGES = 3;
    static constexpr int SEM_COUNT =
        1 + (INPUT_PIPELINE_STAGES + OUTPUT_PIPELINE_STAGES) * 2;

    static_assert(Config::NUM_CONSUMER_WARPS % STAGE_PAGES == 0,
                  "NUM_CONSUMER_WARPS must be divisible by STAGE_PAGES");
    static_assert(RDPW >= 64,
                  "RDPW must be >= 64 to avoid TK swizzle mismatch");
    static_assert(1 + INPUT_PIPELINE_STAGES * STAGE_PAGES <= Config::NUM_PAGES,
                  "Not enough pages for weight pipeline");

    // Scratch layout: OUTPUT_PIPELINE_STAGES * NUM_CONSUMER_WARPS * 16 floats
    static constexpr int SCRATCH_BYTES_PER_WARP  = 16 * sizeof(float);
    static constexpr int SCRATCH_BYTES_PER_STAGE =
        SCRATCH_BYTES_PER_WARP * Config::NUM_CONSUMER_WARPS;
    static_assert(OUTPUT_PIPELINE_STAGES * SCRATCH_BYTES_PER_STAGE
                      <= Config::SCRATCH_BYTES,
                  "SCRATCH_BYTES exceeded");

    // -----------------------------------------------------------------------
    // Parsed instruction
    // -----------------------------------------------------------------------
    struct parsed_instruction {
        int layer_idx, start_block_idx, end_block_idx, blocks_per_expert;

#ifdef MIXTRAL_SMALL_TEST
        // SMALL_TEST: one weight-tile per output block, no inner tiling.
        // iters = total loader/consumer iterations.
        static constexpr int INNER_TILES = 1;
        int iters;
        __device__ inline parsed_instruction(
            typename Config::instruction_t &instr) {
            layer_idx        = instr[1];
            start_block_idx  = instr[2];
            end_block_idx    = instr[3];
            blocks_per_expert = end_block_idx - start_block_idx;
            iters            = blocks_per_expert * Globals::num_experts_per_tok;
        }
#else
        // Full path: 14 tile-batches per output block.
        static constexpr int INNER_TILES = Globals::intermediate_dim / STAGE_COLS;
        // iters_outer: number of (output_block, expert) pairs the consumer
        //   processes (one accumulation + store each).
        // loader_iters: total weight-pipeline iterations (7168).
        int iters_outer, loader_iters;
        __device__ inline parsed_instruction(
            typename Config::instruction_t &instr) {
            layer_idx         = instr[1];
            start_block_idx   = instr[2];
            end_block_idx     = instr[3];
            blocks_per_expert = end_block_idx - start_block_idx;
            iters_outer       = blocks_per_expert * Globals::num_experts_per_tok;
            loader_iters      = iters_outer * INNER_TILES;
        }
#endif

        __device__ inline parsed_instruction(megakernel::state<Config> &s)
            : parsed_instruction(s.instruction()) {}
    };

    // -----------------------------------------------------------------------
    // Semaphore accessors (identical layout to matvec_pipeline)
    // -----------------------------------------------------------------------
    __device__ static inline kittens::semaphore &
    activations_arrived(megakernel::state<Config> &s) {
        return s.semaphores()[0];
    }
    __device__ static inline kittens::semaphore &
    weights_arrived(megakernel::state<Config> &s, int stage) {
        return s.semaphores()[1 + stage];
    }
    __device__ static inline kittens::semaphore &
    weights_finished(megakernel::state<Config> &s, int stage) {
        return s.semaphores()[1 + INPUT_PIPELINE_STAGES + stage];
    }
    __device__ static inline kittens::semaphore &
    outputs_arrived(megakernel::state<Config> &s, int stage) {
        return s.semaphores()[1 + 2 * INPUT_PIPELINE_STAGES + stage];
    }
    __device__ static inline kittens::semaphore &
    outputs_finished(megakernel::state<Config> &s, int stage) {
        return s.semaphores()[1 + 2 * INPUT_PIPELINE_STAGES +
                              OUTPUT_PIPELINE_STAGES + stage];
    }

    // -----------------------------------------------------------------------
    // Page helpers
    // -----------------------------------------------------------------------
    static constexpr int ACTIVATION_PAGE  = 0;
    static constexpr int WEIGHTS_START_PAGE = 1;

    __device__ static inline int get_activation_page(megakernel::state<Config> &s) {
        return s.pid(ACTIVATION_PAGE);
    }
    __device__ static inline int get_weight_page(megakernel::state<Config> &s,
                                                  int stage, int offset) {
        return s.pid(WEIGHTS_START_PAGE + stage * STAGE_PAGES + offset);
    }
    __device__ static inline uint8_t *get_output_start(megakernel::state<Config> &s,
                                                        int stage) {
        return (uint8_t *)s.scratch() + (stage * SCRATCH_BYTES_PER_STAGE);
    }

    // -----------------------------------------------------------------------
    // SMALL_TEST path: delegate entirely to matvec_pipeline<REDUCTION_DIM=1024>
    // -----------------------------------------------------------------------
#ifdef MIXTRAL_SMALL_TEST

    // Reuse the full generic pipeline for SMALL_TEST (REDUCTION_DIM=1024 fits).
    struct pipeline_specifics_small {
        template <int TW>
        static __device__ inline void
        load_iter(megakernel::state<Config> &s, const Globals &g,
                  parsed_instruction &inst, int iter, int col_idx,
                  kittens::st_bf<16, TW> &weight_chunk,
                  kittens::semaphore &sem) {
            int expert_loop    = iter / inst.blocks_per_expert;
            int within_expert  = iter % inst.blocks_per_expert;
            int expert_idx     = g.selected_expert_indices.raw_ptr[expert_loop];
            int depth          = inst.layer_idx * Globals::num_experts + expert_idx;

            kittens::tma::load_async<dim::ROW, cache_policy::EVICT_FIRST>(
                weight_chunk, g.expert_down_weights,
                coord<>{0, depth,
                        (inst.start_block_idx + within_expert) *
                            Globals::matvec_block_size,
                        TW * col_idx},
                sem);
        }

        static __device__ inline void
        store(megakernel::state<Config> &s, const Globals &g,
              parsed_instruction &inst, int output_idx, int output_stage) {

            int expert_loop   = output_idx / inst.blocks_per_expert;
            int within_expert = output_idx % inst.blocks_per_expert;
            int block_idx     = inst.start_block_idx + within_expert;
            int expert_idx    = g.selected_expert_indices.raw_ptr[expert_loop];

            uint8_t *output_scratch_start =
                pipeline::get_output_start(s, output_stage);
            kittens::sv_bf<16> &output_smem_bf =
                *reinterpret_cast<kittens::sv_bf<16> *>(output_scratch_start);

            // SMALL_TEST: RDPW=64 is exactly at the safe swizzle boundary.
            // Use the standard matvec_reduce path.
            kittens::rv_fl<16> output_rv;
            matvec_reduce<Config, kittens::sv_fl<16>, kittens::rv_fl<16>,
                          pipeline::SCRATCH_BYTES_PER_WARP>(
                output_scratch_start, output_rv);

            kittens::warp::sync();

            float score = float(g.selected_expert_scores.raw_ptr[expert_loop]);
            kittens::warp::mul(output_rv, output_rv, score);

            kittens::warp::store(output_smem_bf, output_rv);
            kittens::warp::sync();

            if (kittens::warp::laneid() == 0) {
                s.record(megakernel::TEVENT_OUTPUT_READY);
                kittens::tma::store_add_async<cache_policy::EVICT_LAST>(
                    g.hidden_states, output_smem_bf, {block_idx});
                kittens::tma::store_async_read_wait();
            }
            kittens::warp::sync();
        }
    };

    // SMALL_TEST uses the generic pipeline with REDUCTION_DIM = intermediate_dim (1024).
    using pipeline = matvec_pipeline<Config, Globals, parsed_instruction,
                                     pipeline_specifics_small,
                                     Globals::intermediate_dim>;

#endif  // MIXTRAL_SMALL_TEST

    // -----------------------------------------------------------------------
    // Controller
    // -----------------------------------------------------------------------
    struct controller {
        static __device__ int
        release_lid(const Globals &g,
                    typename Config::instruction_t &instruction, int &query) {
#ifdef MIXTRAL_SMALL_TEST
            return pipeline::release_lid(g, instruction, query);
#else
            // Full path: same page layout as matvec_pipeline.
            // 1 activation page + INPUT_PIPELINE_STAGES * STAGE_PAGES weight pages.
            constexpr int TOTAL_ALLOC =
                1 + INPUT_PIPELINE_STAGES * STAGE_PAGES;
            constexpr int EXTRA = Config::NUM_PAGES - TOTAL_ALLOC;

            if (query < EXTRA)
                return TOTAL_ALLOC + query;

            int inner = query - EXTRA;

            parsed_instruction inst{instruction};
            int iters         = inst.loader_iters;
            int stages_used   = (iters < INPUT_PIPELINE_STAGES)
                                    ? iters
                                    : INPUT_PIPELINE_STAGES;
            int last_stage    = (iters > 0)
                                    ? ((iters - 1) % INPUT_PIPELINE_STAGES)
                                    : 0;

            int order[TOTAL_ALLOC];
            int idx = 0;

            // Unused pipeline stage pages
            for (int s = stages_used; s < INPUT_PIPELINE_STAGES; s++) {
                for (int p = 0; p < STAGE_PAGES; p++) {
                    order[idx++] = WEIGHTS_START_PAGE + s * STAGE_PAGES + p;
                }
            }
            // Activation page
            order[idx++] = ACTIVATION_PAGE;
            // Used stages, last-used stage last
            if (stages_used > 0) {
                for (int i = 1; i <= stages_used; i++) {
                    int s = (last_stage + i) % stages_used;
                    for (int p = 0; p < STAGE_PAGES; p++) {
                        order[idx++] = WEIGHTS_START_PAGE + s * STAGE_PAGES + p;
                    }
                }
            }
            return order[inner];
#endif
        }

        static __device__ int init_semaphores(const Globals &g,
                                              megakernel::state<Config> &s) {
#ifdef MIXTRAL_SMALL_TEST
            return pipeline::init_semaphores(s);
#else
            // Activation semaphore (init count 1: loader signals it once).
            init_semaphore(activations_arrived(s), 1);
            // Weight pipeline: loader signals weights_arrived; each consumer
            // warp arrives at weights_finished (count = NUM_CONSUMER_WARPS).
            for (int i = 0; i < INPUT_PIPELINE_STAGES; i++) {
                init_semaphore(weights_arrived(s, i), 1);
                init_semaphore(weights_finished(s, i),
                               Config::NUM_CONSUMER_WARPS);
            }
            // Output pipeline: NUM_CONSUMER_WARPS consumer warps signal
            // outputs_arrived; storer signals outputs_finished (count=1).
            for (int i = 0; i < OUTPUT_PIPELINE_STAGES; i++) {
                init_semaphore(outputs_arrived(s, i),
                               Config::NUM_CONSUMER_WARPS);
                init_semaphore(outputs_finished(s, i), 1);
            }
            return SEM_COUNT;
#endif
        }
    };

    // -----------------------------------------------------------------------
    // Loader
    // -----------------------------------------------------------------------
    struct loader {
        static __device__ void run(const Globals &g,
                                   megakernel::state<Config> &s) {
#ifdef MIXTRAL_SMALL_TEST
            pipeline::loader_loop(s, g);
#else
            // Full path: streams weight tiles for
            //   loader_iters = blocks_per_expert * num_experts_per_tok * INNER_TILES
            // total iterations through a 6-stage-deep pipeline.
            //
            // Each loader iteration corresponds to one (outer, tile_batch) pair:
            //   outer_iter  = iter / INNER_TILES  (which output_block x expert)
            //   tile_batch  = iter % INNER_TILES  (which 1024-column chunk)
            //
            // We load STAGE_PAGES=2 pages per iteration, covering 1024 columns.
            parsed_instruction inst{s};
            constexpr int INNER_TILES = parsed_instruction::INNER_TILES;

            auto needed_pages =
                1 + min(inst.loader_iters, INPUT_PIPELINE_STAGES) * STAGE_PAGES;

            if (kittens::laneid() == 0) {
                int input_stage = 0;

                for (int iter = 0; iter < inst.loader_iters; iter++) {
                    kittens::wait(
                        weights_finished(s, input_stage),
                        (iter % (2 * INPUT_PIPELINE_STAGES)) <
                            INPUT_PIPELINE_STAGES);

                    int outer_iter   = iter / INNER_TILES;
                    int tile_batch   = iter % INNER_TILES;
                    int expert_loop  = outer_iter / inst.blocks_per_expert;
                    int within_expert = outer_iter % inst.blocks_per_expert;
                    int expert_idx   = g.selected_expert_indices.raw_ptr[expert_loop];
                    int depth        = inst.layer_idx * Globals::num_experts + expert_idx;
                    int start_col    = tile_batch * STAGE_COLS;
                    int row          = (inst.start_block_idx + within_expert) *
                                       Globals::matvec_block_size;

                    // Expect STAGE_PAGES * TILE_WIDTH * 16 * sizeof(bf16) bytes.
                    auto &sem = weights_arrived(s, input_stage);
                    kittens::tma::expect_bytes(
                        sem,
                        sizeof(kittens::bf16) * STAGE_COLS * 16);

                    // Load each page in this stage (each page = 16 x TILE_WIDTH).
#pragma unroll
                    for (int i = 0; i < STAGE_PAGES; i++) {
                        int weight_page = get_weight_page(s, input_stage, i);
                        if (iter < INPUT_PIPELINE_STAGES) {
                            s.wait_page_ready(weight_page);
                        }
                        kittens::st_bf<16, TILE_WIDTH> &weight_chunk =
                            reinterpret_cast<kittens::st_bf<16, TILE_WIDTH> &>(
                                s.pages[weight_page]);

                        if (iter == 0 && i == 0) {
                            s.record(megakernel::TEVENT_FIRST_LOAD);
                        } else if (iter == inst.loader_iters - 1 &&
                                   i == STAGE_PAGES - 1) {
                            s.record(megakernel::TEVENT_LAST_LOAD);
                        }

                        kittens::tma::load_async<dim::ROW, cache_policy::EVICT_FIRST>(
                            weight_chunk, g.expert_down_weights,
                            coord<>{0, depth, row, start_col + i * TILE_WIDTH},
                            sem);
                    }

                    input_stage = (input_stage + 1) % INPUT_PIPELINE_STAGES;
                }
            } else if (kittens::laneid() >= needed_pages &&
                       kittens::laneid() < Config::NUM_PAGES) {
                auto pid = s.pid(kittens::laneid());
                s.wait_page_ready(pid);
                s.finish_page(pid, Config::NUM_CONSUMER_WARPS);
            }
#endif
        }
    };

    // -----------------------------------------------------------------------
    // Launcher
    // -----------------------------------------------------------------------
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

    // -----------------------------------------------------------------------
    // Consumer
    // -----------------------------------------------------------------------
    struct consumer {
        static __device__ void run(const Globals &g,
                                   megakernel::state<Config> &s) {
#ifdef MIXTRAL_SMALL_TEST
            // SMALL_TEST: use generic pipeline consumer.
            // RDPW=64, so matvec() works correctly.
            using sv_t = kittens::sv_bf<pipeline::REDUCTION_DIM_PER_WARP>;
            using rv_t = kittens::rv_fl<pipeline::REDUCTION_DIM_PER_WARP>;
            parsed_instruction inst{s};

            // Wait for RmsRouterUpgate barrier (warp 0, lane 0 spins).
            if (kittens::laneid() == 0 && kittens::warpid() == 0) {
                int activation_page = pipeline::get_activation_page(s);
                s.wait_page_ready(activation_page);

                s.record(megakernel::TEVENT_AT_GMEM_WAIT);
                while (*(volatile int *)&g.Bar[{inst.layer_idx,
                                                prev_opcode - 1, 0}] <
                       EXPECTED_ARRIVAL_COUNT) {
                    __nanosleep(Config::GMEM_SPIN_LOOP_SLEEP_NANOS);
                }
                s.record(megakernel::TEVENT_DONE_GMEM_WAIT);
            }
            kittens::group<Config::NUM_CONSUMER_WARPS>::sync(4);
            asm volatile("fence.acquire.gpu;\n" ::: "memory");

            // Load expert 0 activations for this warp's slice.
            int expert_idx_0 = g.selected_expert_indices.raw_ptr[0];
            sv_t &activations_smem = reinterpret_cast<sv_t *>(
                &pipeline::get_activations(s))[kittens::warpid()];
            kittens::warp::load(
                activations_smem, g.expert_silu_out,
                coord<>{0, 0, expert_idx_0,
                        kittens::warpid() * pipeline::REDUCTION_DIM_PER_WARP});
            kittens::warp::sync();

            rv_t activations_vec;
            kittens::warp::load(activations_vec, activations_smem);
            kittens::warp::sync();

            s.warp_finish_page(pipeline::get_activation_page(s), 1);

            // Custom loop that reloads activations at the expert boundary.
            constexpr int SP = pipeline::STAGE_PAGES;
            constexpr int WPP = Config::NUM_CONSUMER_WARPS / SP;
            int page_index = kittens::warpid() / WPP;

            int input_stage = 0, output_stage = 0;
            for (int i = 0; i < inst.iters; i++) {
                // Reload activations for expert 1 at the boundary.
                if (i == inst.blocks_per_expert) {
                    int expert_idx_1 = g.selected_expert_indices.raw_ptr[1];
                    kittens::warp::load(
                        activations_smem, g.expert_silu_out,
                        coord<>{0, 0, expert_idx_1,
                                kittens::warpid() *
                                    pipeline::REDUCTION_DIM_PER_WARP});
                    kittens::warp::sync();
                    kittens::warp::load(activations_vec, activations_smem);
                    kittens::warp::sync();
                }

                int weight_page = pipeline::get_weight_page(s, input_stage, page_index);
                kittens::wait(
                    pipeline::weights_arrived(s, input_stage),
                    (i % (2 * pipeline::INPUT_PIPELINE_STAGES)) >=
                        pipeline::INPUT_PIPELINE_STAGES);
                kittens::wait(
                    pipeline::outputs_finished(s, output_stage),
                    (i % (2 * pipeline::OUTPUT_PIPELINE_STAGES)) <
                        pipeline::OUTPUT_PIPELINE_STAGES);

                kittens::st_bf<16, pipeline::REDUCTION_DIM_PER_WARP> &weights =
                    reinterpret_cast<
                        kittens::st_bf<16, pipeline::REDUCTION_DIM_PER_WARP> *>(
                        s.pages[weight_page].ptr())[kittens::warpid() % WPP];

                kittens::sv_fl<16> &out_smem =
                    *reinterpret_cast<kittens::sv_fl<16> *>(
                        pipeline::get_output_start(s, output_stage) +
                        (kittens::warpid() * pipeline::SCRATCH_BYTES_PER_WARP));

                if (i == 0) {
                    s.record(megakernel::TEVENT_FIRST_USE);
                } else if (i == inst.iters - 1) {
                    s.record(megakernel::TEVENT_LAST_USE);
                }

                matvec(out_smem, weights, activations_vec);

                kittens::warp::sync();
                kittens::warp::arrive(pipeline::outputs_arrived(s, output_stage));
                kittens::warp::arrive(pipeline::weights_finished(s, input_stage));

                if (i >= inst.iters - pipeline::INPUT_PIPELINE_STAGES) {
#pragma unroll
                    for (int j = 0; j < SP; j++) {
                        s.warp_finish_page(
                            pipeline::get_weight_page(s, input_stage, j), 1);
                    }
                }

                input_stage  = (input_stage  + 1) % pipeline::INPUT_PIPELINE_STAGES;
                output_stage = (output_stage + 1) % pipeline::OUTPUT_PIPELINE_STAGES;
            }

#else  // Full path (intermediate_dim = 14336)
            // ----------------------------------------------------------------
            // Full streaming consumer.
            //
            // Outer loop: iters_outer = blocks_per_expert * num_experts_per_tok
            //   For each outer iter:
            //     - Determine which (output_block, expert) we are.
            //     - At expert boundary: reload activation slices from gmem.
            //     - Inner loop: INNER_TILES tile-batches of 1024 columns.
            //       * Wait for weight pages, load warp's st_bf<16,64> subtile.
            //       * Load activation slice (64 elements) from expert_silu_out.
            //       * matvec -> sv_fl<16> scratch (overwrites warp's scratch).
            //       * Accumulate sv_fl<16> scratch into rv_fl<16> warp_acc.
            //       * Signal weights_finished after matvec.
            //     - Write warp_acc to warp's scratch sv_fl<16>.
            //     - Signal outputs_arrived (all NUM_CONSUMER_WARPS warps).
            //
            // The storer waits on outputs_arrived and reduces warp partials.
            // ----------------------------------------------------------------
            constexpr int INNER_TILES = parsed_instruction::INNER_TILES;

            parsed_instruction inst{s};
            const int warpid = kittens::warpid();

            // Page index within a pipeline stage (0 or 1 for STAGE_PAGES=2).
            // Warp i reads page (i / WARPS_PER_PAGE).
            const int page_index = warpid / WARPS_PER_PAGE;

            // Wait for RmsRouterUpgate barrier (warp 0, lane 0 spins).
            if (kittens::laneid() == 0 && warpid == 0) {
                // No activation page to wait on in the full path
                // (we load activations directly from global memory).
                s.record(megakernel::TEVENT_AT_GMEM_WAIT);
                while (*(volatile int *)&g.Bar[{inst.layer_idx,
                                                prev_opcode - 1, 0}] <
                       EXPECTED_ARRIVAL_COUNT) {
                    __nanosleep(Config::GMEM_SPIN_LOOP_SLEEP_NANOS);
                }
                s.record(megakernel::TEVENT_DONE_GMEM_WAIT);
            }
            kittens::group<Config::NUM_CONSUMER_WARPS>::sync(4);
            asm volatile("fence.acquire.gpu;\n" ::: "memory");

            // Acquire and immediately release the activation page — the full path
            // never loads anything into it, but we must wait_page_ready before
            // finish_page to satisfy the pipeline's page-reuse semaphore protocol.
            {
                int act_page = get_activation_page(s);
                if (kittens::laneid() == 0 && warpid == 0) {
                    s.wait_page_ready(act_page);
                }
                kittens::group<Config::NUM_CONSUMER_WARPS>::sync(5);
                s.warp_finish_page(act_page, 1);
            }

            int input_stage  = 0;
            int output_stage = 0;

            for (int outer = 0; outer < inst.iters_outer; outer++) {
                int expert_loop   = outer / inst.blocks_per_expert;
                int within_expert = outer % inst.blocks_per_expert;
                int block_idx     = inst.start_block_idx + within_expert;
                int expert_idx    = g.selected_expert_indices.raw_ptr[expert_loop];

                // Float32 accumulator for this (output_block, expert).
                kittens::rv_fl<16> warp_acc;
                kittens::warp::zero(warp_acc);

                // Scratch location for this warp (reused across inner iters).
                kittens::sv_fl<16> &warp_scratch =
                    *reinterpret_cast<kittens::sv_fl<16> *>(
                        get_output_start(s, output_stage) +
                        (warpid * SCRATCH_BYTES_PER_WARP));

                for (int t = 0; t < INNER_TILES; t++) {
                    int iter = outer * INNER_TILES + t;

                    // Wait for weight pages for this tile-batch to arrive.
                    kittens::wait(
                        weights_arrived(s, input_stage),
                        (iter % (2 * INPUT_PIPELINE_STAGES)) >=
                            INPUT_PIPELINE_STAGES);

                    // Wait for output scratch to be free (storer has finished
                    // with the previous use of this output stage).
                    // Only wait on the first inner tile of each outer iter.
                    if (t == 0) {
                        kittens::wait(
                            outputs_finished(s, output_stage),
                            (outer % (2 * OUTPUT_PIPELINE_STAGES)) <
                                OUTPUT_PIPELINE_STAGES);
                    }

                    // Identify this warp's st_bf<16, RDPW> subtile within
                    // the stage. Each page holds 16 x TILE_WIDTH=512 elements;
                    // STAGE_PAGES=2 pages cover 16 x 1024 columns.
                    // Warp i maps to page (i/WARPS_PER_PAGE) and sub-tile
                    // offset (i % WARPS_PER_PAGE) within that page.
                    // RDPW = 1024 / NUM_CONSUMER_WARPS = 64 columns per warp.
                    // Within a 512-column page (TILE_WIDTH=512), warp gets
                    // a 64-column sub-tile at offset (i % WARPS_PER_PAGE).
                    int weight_page = get_weight_page(s, input_stage, page_index);
                    kittens::st_bf<16, RDPW> &weights =
                        reinterpret_cast<kittens::st_bf<16, RDPW> *>(
                            s.pages[weight_page].ptr())
                            [warpid % WARPS_PER_PAGE];

                    // Load this warp's RDPW-element activation slice directly
                    // from global memory into registers, bypassing shared memory.
                    // act_col is in element units; coord<rv_fl<RDPW>> expects
                    // column in units of RDPW (unit_coord scales by rv::length).
                    int act_col = t * STAGE_COLS + warpid * RDPW;
                    kittens::rv_fl<RDPW> act_vec;
                    kittens::warp::load(act_vec, g.expert_silu_out,
                                        coord<kittens::rv_fl<RDPW>>{0, 0, expert_idx, act_col / RDPW});

                    // Compute matvec: warp_scratch = weights @ act_vec
                    // This overwrites warp_scratch with the partial sum for
                    // this tile-batch.
                    if (iter == 0 && t == 0) {
                        s.record(megakernel::TEVENT_FIRST_USE);
                    }

                    matvec(warp_scratch, weights, act_vec);
                    // matvec() calls warp::sync() internally.

                    // Accumulate this tile-batch's partial sum into warp_acc.
                    kittens::rv_fl<16> tile_partial;
                    kittens::warp::load(tile_partial, warp_scratch);
                    kittens::warp::add(warp_acc, warp_acc, tile_partial);

                    // Signal that we are done with this weight page stage.
                    kittens::warp::arrive(weights_finished(s, input_stage));

                    // Release weight pages on the last few iterations
                    // (same drain logic as matvec_pipeline::consumer_loop).
                    int total_loader_iters = inst.loader_iters;
                    if (iter >= total_loader_iters - INPUT_PIPELINE_STAGES) {
#pragma unroll
                        for (int j = 0; j < STAGE_PAGES; j++) {
                            s.warp_finish_page(
                                get_weight_page(s, input_stage, j), 1);
                        }
                    }

                    input_stage = (input_stage + 1) % INPUT_PIPELINE_STAGES;
                }  // end inner tile loop

                if (outer == inst.iters_outer - 1) {
                    s.record(megakernel::TEVENT_LAST_USE);
                }

                // Write accumulated result to scratch for the storer.
                // warp_scratch currently holds the last tile-batch's partial;
                // overwrite it with the full accumulation warp_acc.
                kittens::warp::store(warp_scratch, warp_acc);
                kittens::warp::sync();

                // All warps signal that outputs have arrived for this block.
                kittens::warp::arrive(outputs_arrived(s, output_stage));

                output_stage = (output_stage + 1) % OUTPUT_PIPELINE_STAGES;
            }  // end outer loop
#endif  // MIXTRAL_SMALL_TEST
        }
    };

    // -----------------------------------------------------------------------
    // Storer
    // -----------------------------------------------------------------------
    struct storer {
        static __device__ void run(const Globals &g,
                                   megakernel::state<Config> &s) {
#ifdef MIXTRAL_SMALL_TEST
            pipeline::storer_loop(s, g);
            kittens::warp::sync();

            if (kittens::laneid() == 0) {
                parsed_instruction inst{s};
                s.record(megakernel::TEVENT_AT_GMEM_STORE);
                kittens::tma::store_async_wait();
                atomicAdd(&g.Bar[{inst.layer_idx, opcode - 1, 0}],
                          inst.iters);
                s.record(megakernel::TEVENT_DONE_GMEM_STORE);
            }
#else
            // Full path storer:
            //   Iterates iters_outer = blocks_per_expert * num_experts_per_tok times.
            //   For each outer iter:
            //     1. Wait for outputs_arrived (all NUM_CONSUMER_WARPS warps wrote).
            //     2. Reduce warp partials (matvec_reduce over scratch).
            //     3. Multiply by expert score.
            //     4. store_add_async to hidden_states.
            //     5. Signal outputs_finished so the consumer can reuse scratch.
            parsed_instruction inst{s};

            // Reuse the scratch output region directly.
            // Each warp wrote its rv_fl<16> (= 16 floats) to
            //   get_output_start(s, output_stage) + warpid * SCRATCH_BYTES_PER_WARP.
            // The storer reduces these into one sv_bf<16>.

            int output_stage = 0;

            if (kittens::laneid() == 0) {
                s.record(megakernel::TEVENT_AT_GMEM_STORE);
            }

            for (int outer = 0; outer < inst.iters_outer; outer++) {
                int expert_loop   = outer / inst.blocks_per_expert;
                int within_expert = outer % inst.blocks_per_expert;
                int block_idx     = inst.start_block_idx + within_expert;
                int expert_idx    = g.selected_expert_indices.raw_ptr[expert_loop];

                auto &sem = outputs_arrived(s, output_stage);
                bool bit  = (outer % (2 * OUTPUT_PIPELINE_STAGES)) >=
                            OUTPUT_PIPELINE_STAGES;
                kittens::wait(sem, bit);

                if (outer == 0) {
                    s.record(megakernel::TEVENT_FIRST_STORE);
                } else if (outer == inst.iters_outer - 1) {
                    s.record(megakernel::TEVENT_LAST_STORE);
                }

                // Reduce all warp partial sums.
                uint8_t *scratch = get_output_start(s, output_stage);
                kittens::rv_fl<16> output_rv;
                matvec_reduce<Config, kittens::sv_fl<16>, kittens::rv_fl<16>,
                              SCRATCH_BYTES_PER_WARP>(scratch, output_rv);

                // Scale by expert routing score.
                float score = float(g.selected_expert_scores.raw_ptr[expert_loop]);
                kittens::warp::mul(output_rv, output_rv, score);

                // Convert to bf16 and store_add to hidden_states.
                kittens::sv_bf<16> &output_smem_bf =
                    *reinterpret_cast<kittens::sv_bf<16> *>(scratch);
                kittens::warp::store(output_smem_bf, output_rv);
                kittens::warp::sync();

                if (kittens::warp::laneid() == 0) {
                    s.record(megakernel::TEVENT_OUTPUT_READY);
                    kittens::tma::store_add_async<cache_policy::EVICT_LAST>(
                        g.hidden_states, output_smem_bf, {block_idx});
                    kittens::tma::store_async_read_wait();
                }
                kittens::warp::sync();

                // Signal the consumer that this output stage is free.
                kittens::warp::arrive(outputs_finished(s, output_stage));

                output_stage = (output_stage + 1) % OUTPUT_PIPELINE_STAGES;
            }

            kittens::warp::sync();
            if (kittens::laneid() == 0) {
                kittens::tma::store_async_wait();
                // Release fence: all TMA writes must be globally visible
                // before downstream SMs see the barrier increment.
                asm volatile("fence.acq_rel.gpu;\n" ::: "memory");
                atomicAdd(&g.Bar[{inst.layer_idx, opcode - 1, 0}],
                          inst.iters_outer);
                s.record(megakernel::TEVENT_DONE_GMEM_STORE);
            }
#endif
        }
    };
};
