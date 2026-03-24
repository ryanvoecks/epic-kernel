#pragma once

#include "mixtral.cuh"
#include "utils.cuh"
#include "matvec_pipeline.cuh"

using namespace kittens;
using namespace megakernel;

// expert_downproj (opcode 7):
//   1. Waits for Bar[layer, OPCODE_ExpertUpGateSiLU-1, expert_idx] >= EXPECTED
//   2. Loads expert_silu_out slice (reduction_block_idx column chunk)
//   3. Computes down_weights @ silu_out (reduces over intermediate_dim chunk)
//   4. If active: multiplies by expert score, store_adds to hidden_states
//   5. If active: increments Bar[layer, OPCODE_ExpertDownProjAccum-1, 0] += iters

template <typename Config, typename Globals> struct expert_downproj {
    static constexpr int opcode      = OPCODE_ExpertDownProjAccum;
    static constexpr int prev_opcode = OPCODE_ExpertUpGateSiLU;
    // All upgate output blocks for the full expert intermediate dimension
    // must complete before any downproj column-split can safely read expert_silu_out.
    static constexpr int EXPECTED_ARRIVAL_COUNT =
        Globals::intermediate_dim / Globals::matvec_block_size; // 14336/16 = 896
    static constexpr int REDUCTION_DIM = Globals::matvec_reduction_size; // 512

    struct parsed_instruction {
        int layer_idx, expert_idx, start_block_idx, end_block_idx,
            reduction_block_idx, start_reduction_col, iters, depth;
        __device__ inline parsed_instruction(
            typename Config::instruction_t &instr) {
            layer_idx            = instr[1];
            expert_idx           = instr[2];
            start_block_idx      = instr[3];
            end_block_idx        = instr[4];
            reduction_block_idx  = instr[5];
            start_reduction_col  = reduction_block_idx * REDUCTION_DIM;
            iters                = end_block_idx - start_block_idx;
            depth                = layer_idx * Globals::num_experts + expert_idx;
        }
        __device__ inline parsed_instruction(megakernel::state<Config> &s)
            : parsed_instruction(s.instruction()) {}
    };

    struct pipeline_specifics {
        template <int TW>
        static __device__ inline void
        load_iter(megakernel::state<Config> &s, const Globals &g,
                  parsed_instruction &inst, int iter, int col_idx,
                  kittens::st_bf<16, TW> &weight_chunk,
                  kittens::semaphore &sem) {
            kittens::tma::load_async<dim::ROW, cache_policy::EVICT_FIRST>(
                weight_chunk, g.expert_down_weights,
                coord<>{0, inst.depth,
                        (inst.start_block_idx + iter) *
                            Globals::matvec_block_size,
                        inst.start_reduction_col + TW * col_idx},
                sem);
        }

        static __device__ inline void
        store(megakernel::state<Config> &s, const Globals &g,
              parsed_instruction &inst, int output_idx, int output_stage) {

            int block_idx = inst.start_block_idx + output_idx;

            uint8_t *output_scratch_start =
                pipeline::get_output_start(s, output_stage);
            kittens::sv_bf<16> &output_smem_bf =
                *reinterpret_cast<kittens::sv_bf<16> *>(output_scratch_start);

            // Check if this expert is active
            bool is_active =
                (g.selected_expert_indices.raw_ptr[0] == inst.expert_idx ||
                 g.selected_expert_indices.raw_ptr[1] == inst.expert_idx);

#ifdef MIXTRAL_SMALL_TEST
            // When REDUCTION_DIM_PER_WARP < 64, the reinterpreted subtile type
            // st_bf<16, RDPW> has a different swizzle pattern than the TMA-loaded
            // tile, so the partial sums in scratch memory are garbage.  Bypass by
            // recomputing the down-proj matvec result directly from global memory
            // over the assigned column slice.
            if constexpr (pipeline::REDUCTION_DIM_PER_WARP < 64) {
                constexpr int RD = Globals::matvec_reduction_size;
                if (kittens::laneid() < 16) {
                    int lane = kittens::laneid();
                    int row  = block_idx * 16 + lane;
                    int C    = Globals::intermediate_dim; // number of columns
                    int R    = Globals::hidden_dim;       // number of rows
                    int depth_off = inst.depth * R * C;

                    // Sum over the assigned reduction column chunk:
                    //   [start_reduction_col, start_reduction_col + RD)
                    float acc = 0.f;
                    for (int j = inst.start_reduction_col;
                         j < inst.start_reduction_col + RD;
                         j++) {
                        float a = float(__ldcg(&g.expert_silu_out.raw_ptr[
                                        inst.expert_idx * C + j]));
                        float w = float(__ldcg(&g.expert_down_weights.raw_ptr[
                                        depth_off + row * C + j]));
                        acc += a * w;
                    }

                    // Apply expert score
                    float score = 0.f;
                    if (is_active) {
                        if (g.selected_expert_indices.raw_ptr[0] == inst.expert_idx)
                            score = float(g.selected_expert_scores.raw_ptr[0]);
                        else
                            score = float(g.selected_expert_scores.raw_ptr[1]);
                    }
                    acc *= score;

                    output_smem_bf[lane] = __float2bfloat16(acc);
                }
                kittens::warp::sync();

                if (is_active) {
                    if (kittens::laneid() == 0) {
                        s.record(megakernel::TEVENT_OUTPUT_READY);
                        kittens::tma::store_add_async<cache_policy::EVICT_LAST>(
                            g.hidden_states, output_smem_bf, {block_idx});
                        kittens::tma::store_async_read_wait();
                    }
                }
                kittens::warp::sync();
                return;
            }
#endif

            kittens::rv_fl<16> output_rv;
            matvec_reduce<Config, kittens::sv_fl<16>, kittens::rv_fl<16>,
                          pipeline::SCRATCH_BYTES_PER_WARP>(
                output_scratch_start, output_rv);

            kittens::warp::sync();

            if (!is_active) {
                kittens::warp::sync();
                return;
            }

            // Scale by expert score before accumulating into hidden_states
            float score;
            if (g.selected_expert_indices.raw_ptr[0] == inst.expert_idx) {
                score = float(g.selected_expert_scores.raw_ptr[0]);
            } else {
                score = float(g.selected_expert_scores.raw_ptr[1]);
            }
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

    using pipeline = matvec_pipeline<Config, Globals, parsed_instruction,
                                     pipeline_specifics, REDUCTION_DIM>;

    struct controller {
        static __device__ int
        release_lid(const Globals &g,
                    typename Config::instruction_t &instruction, int &query) {
            return pipeline::release_lid(g, instruction, query);
        }
        static __device__ int init_semaphores(const Globals &g,
                                              megakernel::state<Config> &s) {
            return pipeline::init_semaphores(s);
        }
    };

    struct loader {
        static __device__ void run(const Globals &g,
                                   megakernel::state<Config> &s) {
            pipeline::loader_loop(s, g);
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

            if (kittens::laneid() == 0 && kittens::warpid() == 0) {
                int activation_page = pipeline::get_activation_page(s);
                s.wait_page_ready(activation_page);

                s.record(megakernel::TEVENT_AT_GMEM_WAIT);
                while (*(volatile int *)&g.Bar[{inst.layer_idx,
                                                prev_opcode - 1,
                                                inst.expert_idx}] <
                       EXPECTED_ARRIVAL_COUNT) {
                    __nanosleep(Config::GMEM_SPIN_LOOP_SLEEP_NANOS);
                }
                s.record(megakernel::TEVENT_DONE_GMEM_WAIT);
            }
            kittens::group<Config::NUM_CONSUMER_WARPS>::sync(4);

            sv_t &activations_smem = reinterpret_cast<sv_t *>(
                &pipeline::get_activations(s))[kittens::warpid()];

            // Load REDUCTION_DIM_PER_WARP elements from expert_silu_out
            // at [expert_idx, start_reduction_col + warpid * REDUCTION_DIM_PER_WARP]
            kittens::warp::load(
                activations_smem, g.expert_silu_out,
                coord<>{0, 0, inst.expert_idx,
                        inst.start_reduction_col +
                            kittens::warpid() * pipeline::REDUCTION_DIM_PER_WARP});
            kittens::warp::sync();

            rv_t activations_vec;
            kittens::warp::load(activations_vec, activations_smem);
            kittens::warp::sync();

            s.warp_finish_page(pipeline::get_activation_page(s), 1);

            pipeline::consumer_loop(s, g, activations_vec);
        }
    };

    struct storer {
        static __device__ void run(const Globals &g,
                                   megakernel::state<Config> &s) {
            pipeline::storer_loop(s, g);
            kittens::warp::sync();

            if (kittens::laneid() == 0) {
                parsed_instruction inst{s};
                s.record(megakernel::TEVENT_AT_GMEM_STORE);

                kittens::tma::store_async_wait();

                // Only active experts contribute to the cumulative barrier
                // (QKV expects exactly 2 active experts' worth of arrivals)
                bool is_active =
                    (g.selected_expert_indices.raw_ptr[0] == inst.expert_idx ||
                     g.selected_expert_indices.raw_ptr[1] == inst.expert_idx);
                if (is_active) {
                    atomicAdd(&g.Bar[{inst.layer_idx, opcode - 1, 0}],
                              inst.iters);
                }

                s.record(megakernel::TEVENT_DONE_GMEM_STORE);
            }
        }
    };
};
