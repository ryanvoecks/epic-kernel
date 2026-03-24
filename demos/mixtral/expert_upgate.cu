#pragma once

#include "mixtral.cuh"
#include "utils.cuh"
#include "matvec_pipeline.cuh"

using namespace kittens;
using namespace megakernel;

// expert_upgate (opcode 6):
//   1. Waits for Router barrier (Bar[layer, OPCODE_MoE_Router-1, 0] >= 1)
//   2. Loads normed hidden from router_normed_hidden
//   3. Computes gate * up matVec for assigned output blocks
//   4. Applies SiLU(gate) * up
//   5. Writes to expert_silu_out[{0,0,expert_idx,block_idx}] (active experts only)
//   6. Increments Bar[layer, OPCODE_ExpertUpGateSiLU-1, expert_idx] += num_blocks (always)

template <typename Config, typename Globals> struct expert_upgate {
    static constexpr int opcode = OPCODE_ExpertUpGateSiLU;
    // Router signals exactly 1 arrival
    static constexpr int EXPECTED_ROUTER_COUNT = 1;

    struct parsed_instruction {
        int layer_idx, expert_idx, start_block_idx, num_blocks, iters, depth;
        __device__ inline parsed_instruction(
            typename Config::instruction_t &instr) {
            layer_idx       = instr[1];
            expert_idx      = instr[2];
            start_block_idx = instr[3];
            num_blocks      = instr[4];
            iters           = 2 * num_blocks; // gate + up interleaved
            depth           = layer_idx * Globals::num_experts + expert_idx;
        }
        __device__ inline parsed_instruction(megakernel::state<Config> &s)
            : parsed_instruction(s.instruction()) {}
    };

    struct pipeline_specifics {
        // gmem_wait is handled manually in the consumer; this is a no-op here
        static __device__ inline void gmem_wait(const Globals &g,
                                                megakernel::state<Config> &s) {}

        template <int TW>
        static __device__ inline void
        load_iter(megakernel::state<Config> &s, const Globals &g,
                  parsed_instruction &inst, int iter, int col_idx,
                  kittens::st_bf<16, TW> &weight_chunk,
                  kittens::semaphore &sem) {
            auto block_idx = inst.start_block_idx + iter / 2;
            if (iter % 2 == 0) {
                // even iter: up weights
                kittens::tma::load_async<dim::ROW, cache_policy::EVICT_FIRST>(
                    weight_chunk, g.expert_up_weights,
                    {0, inst.depth, block_idx, col_idx}, sem);
            } else {
                // odd iter: gate weights
                kittens::tma::load_async<dim::ROW, cache_policy::EVICT_FIRST>(
                    weight_chunk, g.expert_gate_weights,
                    {0, inst.depth, block_idx, col_idx}, sem);
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

#ifdef MIXTRAL_SMALL_TEST
            // Consumer already wrote correct values to expert_silu_out directly;
            // skip matvec_reduce + TMA store (RDPW<64 swizzle mismatch would crash).
            if constexpr (pipeline::REDUCTION_DIM_PER_WARP < 64) {
                return;
            }
#endif

            auto true_output_idx   = output_idx / 2;
            auto prev_output_idx   = output_idx - 1;
            auto prev_output_stage = prev_output_idx % pipeline::OUTPUT_PIPELINE_STAGES;

            int block_idx = inst.start_block_idx + true_output_idx;

            uint8_t *output_scratch_start =
                pipeline::get_output_start(s, output_stage);
            uint8_t *prev_output_scratch_start =
                pipeline::get_output_start(s, prev_output_stage);

            kittens::sv_bf<16> &out_smem =
                *reinterpret_cast<kittens::sv_bf<16> *>(output_scratch_start);

            kittens::rv_fl<16> up_out, gate_out, gate_scratch;

            // Reduce up (prev stage) and gate (current stage)
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

            // Only store to expert_silu_out if this expert is active
            bool is_active =
                (g.selected_expert_indices.raw_ptr[0] == inst.expert_idx ||
                 g.selected_expert_indices.raw_ptr[1] == inst.expert_idx);

            if (is_active) {
                if (kittens::laneid() == 0) {
                    s.record(megakernel::TEVENT_OUTPUT_READY);
                    kittens::tma::store_async<cache_policy::EVICT_LAST>(
                        g.expert_silu_out, out_smem,
                        {0, 0, inst.expert_idx, block_idx});
                    kittens::tma::store_async_read_wait();
                }
            }

            kittens::warp::sync();
        }
    };

    using pipeline = matvec_pipeline<Config, Globals, parsed_instruction,
                                     pipeline_specifics>;

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
                                                OPCODE_MoE_Router - 1, 0}] <
                       EXPECTED_ROUTER_COUNT) {
                    __nanosleep(Config::GMEM_SPIN_LOOP_SLEEP_NANOS);
                }
                s.record(megakernel::TEVENT_DONE_GMEM_WAIT);
            }
            kittens::group<Config::NUM_CONSUMER_WARPS>::sync(4);

            sv_t &activations_smem = reinterpret_cast<sv_t *>(
                &pipeline::get_activations(s))[kittens::warpid()];

            // Load this warp's chunk of router_normed_hidden.
            // coord<>{N} uses element offset N in the last dimension.
            // warpid=k → element offset k*RDPW → loads elements [k*RDPW..(k+1)*RDPW-1].
            kittens::warp::load(activations_smem, g.router_normed_hidden,
                                kittens::coord<>{kittens::warpid() * pipeline::REDUCTION_DIM_PER_WARP});
            kittens::warp::sync();

            rv_t activations_vec;
            kittens::warp::load(activations_vec, activations_smem);
            kittens::warp::sync();

            s.warp_finish_page(pipeline::get_activation_page(s), 1);

#ifdef MIXTRAL_SMALL_TEST
            // When REDUCTION_DIM_PER_WARP < 64, the st_bf<16, RDPW> tile has a
            // different swizzle pattern than the TMA-loaded st_bf<16, TILE_WIDTH>
            // page, causing matvec() to crash via WGMMA hardware fault on Blackwell.
            // Bypass: warp 0 computes all output blocks using activations from smem
            // (loaded correctly via __ldg-based warp::load earlier).
            if constexpr (pipeline::REDUCTION_DIM_PER_WARP < 64) {
                // Sync to ensure all warps have written their smem activation chunks
                // before warp 0 reads them. (The sync(4) at line 189 precedes the
                // warp::load calls, so we need a fresh barrier here.)
                kittens::group<Config::NUM_CONSUMER_WARPS>::sync(2);

                if (kittens::warpid() == 0) {
                    int lane = kittens::laneid();
                    if (lane < 16) {
                        // All 512 activations are packed contiguously in smem:
                        // [warp0_chunk(32 bf16), warp1_chunk(32 bf16), ..., warp15_chunk(32 bf16)]
                        // Using smem avoids L1 cache staleness for router_normed_hidden
                        // (warp::load used __ldg which reads from L2, correct post-router data).
                        __nv_bfloat16 *all_act = reinterpret_cast<__nv_bfloat16 *>(
                            &pipeline::get_activations(s));

                        for (int b = 0; b < inst.num_blocks; b++) {
                            int block_idx = inst.start_block_idx + b;
                            size_t row = (size_t)block_idx * 16 + lane;
                            size_t C   = Globals::hidden_dim;
                            size_t R   = Globals::intermediate_dim;
                            size_t depth_off = (size_t)inst.depth * R * C;

                            float up_acc = 0.f, gate_acc = 0.f;
                            for (int j = 0; j < Globals::hidden_dim; j++) {
                                float a  = float(all_act[j]);  // smem - correct data
                                up_acc   += a * float(__ldcg(&g.expert_up_weights.raw_ptr[depth_off + row * C + j]));
                                gate_acc += a * float(__ldcg(&g.expert_gate_weights.raw_ptr[depth_off + row * C + j]));
                            }
                            float silu_gate = gate_acc / (1.f + __expf(-gate_acc));
                            float result    = silu_gate * up_acc;

                            int32_t sei0 = __ldcg(&g.selected_expert_indices.raw_ptr[0]);
                            int32_t sei1 = __ldcg(&g.selected_expert_indices.raw_ptr[1]);
                            bool is_active = (sei0 == inst.expert_idx || sei1 == inst.expert_idx);

                            if (lane == 0 && b == 0 && inst.expert_idx < 4) {
                                printf("DBG upgate SM=%d exp=%d blk=%d sei=[%d,%d] is_active=%d act0=%f up_acc=%f result=%f\n",
                                    (int)blockIdx.x, inst.expert_idx, block_idx,
                                    sei0, sei1, (int)is_active,
                                    float(all_act[0]), up_acc, result);
                            }

                            if (is_active) {
                                size_t out_idx = (size_t)inst.expert_idx * R + row;
                                g.expert_silu_out.raw_ptr[out_idx] = __float2bfloat16(result);
                            }
                        }
                    }
                }
                kittens::group<Config::NUM_CONSUMER_WARPS>::sync(4);

                // For each pipeline stage the loader uses: signal weights_finished
                // (unblocks loader), wait for the page to be loaded (weights_arrived),
                // then release the weight page so the page pool is not exhausted.
                // This must be done in stage order so the loader can make progress.
                int stages_used = (inst.iters < pipeline::INPUT_PIPELINE_STAGES)
                                      ? inst.iters
                                      : pipeline::INPUT_PIPELINE_STAGES;
                for (int i = 0; i < stages_used; i++) {
                    // Unblock loader for stage i
                    kittens::warp::arrive(pipeline::weights_finished(s, i));
                    // Wait for loader to have loaded stage i (TMA complete)
                    kittens::wait(pipeline::weights_arrived(s, i), 0);
                    // Release the page(s) for stage i
                    for (int p = 0; p < pipeline::STAGE_PAGES; p++) {
                        s.warp_finish_page(pipeline::get_weight_page(s, i, p), 1);
                    }
                }

                // Signal outputs_arrived for all iters so storer_loop can proceed
                // (storer calls store() which returns early in this bypass path).
                for (int i = 0; i < inst.iters; i++) {
                    kittens::warp::arrive(
                        pipeline::outputs_arrived(s, i % pipeline::OUTPUT_PIPELINE_STAGES));
                }
                return;
            }
#endif

            pipeline::consumer_loop(s, g, activations_vec);
        }
    };

    struct storer {
        static __device__ void run(const Globals &g,
                                   megakernel::state<Config> &s) {
            pipeline::storer_loop<2>(s, g);
            kittens::warp::sync();

            if (kittens::laneid() == 0) {
                parsed_instruction inst{s};
                s.record(megakernel::TEVENT_AT_GMEM_STORE);
#ifndef MIXTRAL_SMALL_TEST
                kittens::tma::store_async_wait();
#endif
                // Always increment barrier (even for inactive experts),
                // so expert_downproj can detect that upgate is done.
                atomicAdd(&g.Bar[{inst.layer_idx, opcode - 1, inst.expert_idx}],
                          inst.num_blocks);
                s.record(megakernel::TEVENT_DONE_GMEM_STORE);
            }
        }
    };
};
