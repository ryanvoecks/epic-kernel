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

            // Load this warp's chunk of router_normed_hidden
            // coord<>{} = default_type, no tile-size scaling; element offset = warpid * RDPW
            kittens::warp::load(activations_smem, g.router_normed_hidden,
                                kittens::coord<>{kittens::warpid() * pipeline::REDUCTION_DIM_PER_WARP});
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
            pipeline::storer_loop<2>(s, g);
            kittens::warp::sync();

            if (kittens::laneid() == 0) {
                parsed_instruction inst{s};
                s.record(megakernel::TEVENT_AT_GMEM_STORE);
                kittens::tma::store_async_wait();
                // Always increment barrier (even for inactive experts),
                // so expert_downproj can detect that upgate is done.
                atomicAdd(&g.Bar[{inst.layer_idx, opcode - 1, inst.expert_idx}],
                          inst.num_blocks);
                s.record(megakernel::TEVENT_DONE_GMEM_STORE);
            }
        }
    };
};
