#pragma once

#include "mixtral.cuh"
#include "utils.cuh"
#include "matvec_pipeline.cuh"

using namespace kittens;
using namespace megakernel;

template <typename Config, typename Globals> struct rms_lm_head {
    static constexpr int opcode = OPCODE_RMS_LM_Head;
    // Wait for ALL expert downproj blocks to finish on the last layer.
    static constexpr int EXPECTED_ARRIVAL_COUNT = Globals::qkv_expected_arrivals;

    // Use half hidden_dim for REDUCTION_DIM to get 3-stage pipelining.
    // Each column-split instruction reduces over 2048 columns and uses
    // store_add_async to accumulate partial logits.
    static constexpr int REDUCTION_DIM = Globals::hidden_dim / 2;

    struct parsed_instruction {
        int start_block_idx, end_block_idx, iters;
        int reduction_block_idx, start_reduction_col;
        __device__ inline parsed_instruction(
            typename Config::instruction_t &instruction) {
            start_block_idx = instruction[1];
            end_block_idx = instruction[2];
            iters = end_block_idx - start_block_idx;
            reduction_block_idx = instruction[3];
            start_reduction_col = reduction_block_idx * REDUCTION_DIM;
        }
        __device__ inline parsed_instruction(megakernel::state<Config> &s)
            : parsed_instruction(s.instruction()) {}
    };

    struct pipeline_specifics {
        static __device__ inline void gmem_wait(const Globals &g,
                                                megakernel::state<Config> &s) {
            parsed_instruction inst{s};
            while (*(volatile int *)&g.Bar[{Globals::num_layers - 1,
                                            OPCODE_ExpertDownProjFused - 1, 0}] <
                   EXPECTED_ARRIVAL_COUNT) {
                __nanosleep(Config::GMEM_SPIN_LOOP_SLEEP_NANOS);
            }
        }

        template <int TW>
        static __device__ inline void
        load_iter(megakernel::state<Config> &s, const Globals &g,
                  parsed_instruction &inst, int iter, int col_idx,
                  kittens::st_bf<16, TW> &weight_chunk,
                  kittens::semaphore &sem) {
            auto block_idx = inst.start_block_idx + iter;
            // Column offset: start_reduction_col maps to TMA tile index.
            int col_coord = inst.start_reduction_col / TW + col_idx;
            kittens::tma::load_async<dim::ROW, cache_policy::EVICT_FIRST>(
                weight_chunk, g.lm_head_weights, {block_idx, col_coord}, sem);
        }

        static __device__ inline void store(megakernel::state<Config> &s,
                                            const Globals &g,
                                            parsed_instruction &inst,
                                            int output_idx, int output_stage) {

            int block_idx = inst.start_block_idx + output_idx;

            uint8_t *output_scratch_start =
                pipeline::get_output_start(s, output_stage);
            kittens::sv_bf<16> &logits_smem_bf =
                *reinterpret_cast<kittens::sv_bf<16> *>(output_scratch_start);

            kittens::rv_fl<16> logits_rv;
            matvec_reduce<Config, kittens::sv_fl<16>, kittens::rv_fl<16>,
                          pipeline::SCRATCH_BYTES_PER_WARP>(
                output_scratch_start, logits_rv);

            kittens::warp::sync();
            kittens::warp::store(logits_smem_bf, logits_rv);
            kittens::warp::sync();

            if (kittens::warp::laneid() == 0) {
                s.record(megakernel::TEVENT_OUTPUT_READY);

                kittens::tma::store_add_async<cache_policy::EVICT_LAST>(
                    g.logits, logits_smem_bf, {0, 0, 0, block_idx});
                kittens::tma::store_async_read_wait();
            }

            kittens::warp::sync();
        }
    };

    using pipeline =
        matvec_pipeline<Config, Globals, parsed_instruction,
                        pipeline_specifics, REDUCTION_DIM>;

    // Semaphore for RMS norm weights loaded by loader (full path only)
    __device__ static inline kittens::semaphore &
    rms_weights_arrived(megakernel::state<Config> &s) {
        return s.semaphores()[pipeline::SEM_COUNT];
    }

    static constexpr int SEM_COUNT = pipeline::SEM_COUNT + 1;

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
            return SEM_COUNT;
        }
    };

    struct loader {
        static __device__ void run(const Globals &g,
                                   megakernel::state<Config> &s) {
            parsed_instruction inst{s};

            // Load RMS norm weights into the activation page
            if (kittens::laneid() == 0) {
                int activation_page = pipeline::get_activation_page(s);
                s.wait_page_ready(activation_page);

                auto &rms_smem = *reinterpret_cast<kittens::sv_bf<Globals::hidden_dim> *>(
                    s.pages[activation_page].ptr(sizeof(kittens::sv_bf<Globals::hidden_dim>)));

                auto &sem = rms_weights_arrived(s);
                kittens::tma::expect(sem, rms_smem);
                kittens::tma::load_async<cache_policy::EVICT_LAST>(
                    rms_smem, g.lm_head_norm_weights, {0, 0}, sem);
            }

            // Then run the standard matvec pipeline loader for weight pages
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

            // ---- Phase 1: Wait for downproj barrier + RMS norm ----

            if (kittens::warpid() == 0 && kittens::laneid() == 0) {
                int activation_page = pipeline::get_activation_page(s);
                s.wait_page_ready(activation_page);

                s.record(megakernel::TEVENT_AT_GMEM_WAIT);
                while (*(volatile int *)&g.Bar[{Globals::num_layers - 1,
                                                OPCODE_ExpertDownProjFused - 1, 0}] <
                       EXPECTED_ARRIVAL_COUNT) {
                    __nanosleep(Config::GMEM_SPIN_LOOP_SLEEP_NANOS);
                }
                s.record(megakernel::TEVENT_DONE_GMEM_WAIT);
            }
            kittens::group<Config::NUM_CONSUMER_WARPS>::sync(3);

            // Wait for RMS norm weights
            kittens::wait(rms_weights_arrived(s), 0);

            int activation_page = pipeline::get_activation_page(s);

            // RMS norm using SMEM pipeline (same as upgate Phase 1)
            constexpr int CHUNK = Globals::hidden_dim / Config::NUM_CONSUMER_WARPS;
            using sv_chunk = kittens::sv_bf<CHUNK>;

            sv_chunk &act_smem = reinterpret_cast<sv_chunk *>(
                s.pages[activation_page].ptr())[kittens::warpid()];
            sv_chunk &nw_smem = reinterpret_cast<sv_chunk *>(
                s.pages[activation_page].ptr(sizeof(kittens::sv_bf<Globals::hidden_dim>)))[kittens::warpid()];

            kittens::warp::load(act_smem, g.hidden_states,
                                kittens::coord<>{kittens::warpid() * CHUNK});
            kittens::warp::sync();

            auto activations_vec_full = rms_norm<Config, Globals>(
                nw_smem, act_smem, g.rms_norm_eps, s.scratch());
            kittens::warp::sync();

            // Write normed activations to router_normed_hidden (reused as scratch)
            {
                sv_chunk &tmp_smem = act_smem;
                kittens::warp::store(tmp_smem, activations_vec_full);
                kittens::warp::sync();
                kittens::warp::store(g.router_normed_hidden, tmp_smem,
                                     kittens::coord<>{kittens::warpid() * CHUNK});
            }

            kittens::group<Config::NUM_CONSUMER_WARPS>::sync(1);

            // Release activation page
            s.warp_finish_page(activation_page, 1);

            // ---- Phase 2: Flat matvec pipeline with REDUCTION_DIM=2048 ----
            // Load this column split's activation slice from router_normed_hidden
            sv_t &act_smem_pipe = reinterpret_cast<sv_t *>(
                &pipeline::get_activations(s))[kittens::warpid()];

            kittens::warp::load(act_smem_pipe, g.router_normed_hidden,
                                kittens::coord<>{inst.start_reduction_col +
                                                 kittens::warpid() * pipeline::REDUCTION_DIM_PER_WARP});
            kittens::warp::sync();

            rv_t act_vec_pipe;
            kittens::warp::load(act_vec_pipe, act_smem_pipe);
            kittens::warp::sync();

            pipeline::consumer_loop(s, g, act_vec_pipe);
        }
    };
    struct storer {
        static __device__ void run(const Globals &g,
                                   megakernel::state<Config> &s) {
            pipeline::storer_loop(s, g);
        }
    };
};
