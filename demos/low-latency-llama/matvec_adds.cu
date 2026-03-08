#pragma once

#include "llama.cuh"
#include "utils.cuh"
#include "matvec_pipeline.cuh"

using namespace kittens;
using namespace megakernel;

// MatVecAddOp is parameterized with _REDUCTION_DIM to control the pipeline's
// column-reduction width. For o_proj: _REDUCTION_DIM = hidden_dim = 3072
// (default). For downproj: _REDUCTION_DIM = downproj_reduction_chunk_size.
template <int _EXPECTED_ARRIVAL_COUNT, auto WeightsPtr,
          auto InputActivationsPtr, auto OutputActivationsPtr, int _opcode,
          int _prev_opcode = 0,
          typename Config = default_config,
          typename Globals = llama_globals,
          int _REDUCTION_DIM = Globals::hidden_dim>
struct MatVecAddOp {
    static constexpr int opcode = _opcode;
    static constexpr int prev_opcode = _prev_opcode;
    static constexpr int EXPECTED_ARRIVAL_COUNT = _EXPECTED_ARRIVAL_COUNT;
    static constexpr int REDUCTION_DIM = _REDUCTION_DIM;

    struct parsed_instruction {
        int layer, start_block_idx, end_block_idx, reduction_block_idx,
            start_reduction_col, iters;
        __device__ inline parsed_instruction(
            typename Config::instruction_t &instruction) {
            layer = instruction[1];
            start_block_idx = instruction[2];
            end_block_idx = instruction[3];
            reduction_block_idx = instruction[4];
            // Use REDUCTION_DIM (not hidden_dim) to compute the start column.
            // For o_proj: REDUCTION_DIM=3072, reduction_block_idx=0 → col=0.
            // For downproj: REDUCTION_DIM=downproj_reduction_chunk_size, reduction_block_idx=0..N → col=0..N*chunk.
            start_reduction_col = reduction_block_idx * REDUCTION_DIM;
            iters = end_block_idx - start_block_idx;
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
                weight_chunk, g.*WeightsPtr,
                coord<>{inst.layer,
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

            kittens::rv_fl<16> output_rv;
            matvec_reduce<Config, kittens::sv_fl<16>, kittens::rv_fl<16>,
                          pipeline::SCRATCH_BYTES_PER_WARP>(
                output_scratch_start, output_rv);

            kittens::warp::sync();
            kittens::warp::store(output_smem_bf, output_rv);
            kittens::warp::sync();

            if (kittens::warp::laneid() == 0) {
                auto &OutputActivations = g.*OutputActivationsPtr;
                kittens::tma::store_add_async<cache_policy::EVICT_LAST>(
                    OutputActivations, output_smem_bf, {block_idx});
                kittens::tma::store_async_read_wait();
            }

            kittens::warp::sync();
        }
    };

    // The pipeline uses REDUCTION_DIM (not always hidden_dim) for its
    // STAGE_PAGES and INPUT_PIPELINE_STAGES calculations.
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
        static __device__ void run(const Globals &g, megakernel::state<Config> &s) {
            pipeline::loader_loop(s, g);
        }
    };
    struct launcher {
        static __device__ void run(const Globals &g, megakernel::state<Config> &s) {
            if (kittens::laneid() == 0) {
#ifdef KITTENS_BLACKWELL
                s.wait_tensor_ready();
                arrive(s.tensor_finished, Config::NUM_CONSUMER_WARPS);
#endif
            }
        }
    };
    struct consumer {
        static __device__ void run(const Globals &g, megakernel::state<Config> &s) {

            using sv_t = kittens::sv_bf<pipeline::REDUCTION_DIM_PER_WARP>;
            using rv_t = kittens::rv_fl<pipeline::REDUCTION_DIM_PER_WARP>;
            parsed_instruction inst{s};

            if (kittens::laneid() == 0 && kittens::warpid() == 0) {

                int activation_page = pipeline::get_activation_page(s);
                s.wait_page_ready(activation_page);

                s.record(megakernel::TEVENT_AT_GMEM_WAIT);
                while (*(volatile int *)&g.Bar[{inst.layer, prev_opcode - 1,
                                                inst.reduction_block_idx}] <
                       EXPECTED_ARRIVAL_COUNT) {
                    __nanosleep(Config::GMEM_SPIN_LOOP_SLEEP_NANOS);
                }
                s.record(megakernel::TEVENT_DONE_GMEM_WAIT);

                auto &activations = pipeline::get_activations(s);
                auto &InputActivations = g.*InputActivationsPtr;
            }
            group<Config::NUM_CONSUMER_WARPS>::sync(4);

            sv_t &activations_smem = reinterpret_cast<sv_t *>(
                &pipeline::get_activations(s))[kittens::warpid()];

            // Load REDUCTION_DIM_PER_WARP elements starting from
            // start_reduction_col offset.
            kittens::warp::load(activations_smem, g.*InputActivationsPtr,
                       coord<>{inst.start_reduction_col +
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
        static __device__ void run(const Globals &g, megakernel::state<Config> &s) {
            pipeline::storer_loop(s, g);
            kittens::warp::sync();

            if (kittens::laneid() == 0) {
                s.record(megakernel::TEVENT_AT_GMEM_STORE);
                parsed_instruction inst{s};

                kittens::tma::store_async_wait();

                atomicAdd(&g.Bar[{inst.layer, opcode - 1, 0}], inst.iters);
                s.record(megakernel::TEVENT_DONE_GMEM_STORE);
            }
        }
    };
};

// downproj: reduces over intermediate_dim via col-splits of downproj_reduction_chunk_size.
// REDUCTION_DIM=downproj_reduction_chunk_size.
// Each col-split covers downproj_reduction_chunk_size/matvec_block_size upgate blocks.
// EXPECTED = downproj_reduction_chunk_size / matvec_block_size.
template <typename Config, typename Globals>
struct downproj : MatVecAddOp<
    Globals::downproj_reduction_chunk_size / Globals::matvec_block_size,
    &Globals::down_weights, &Globals::silu_out,
    &Globals::hidden_states, OPCODE_DownProjResidual,
    OPCODE_DownProjResidual - 1, Config, Globals,
    Globals::downproj_reduction_chunk_size
> {};

// o_proj: reduces over hidden_dim. REDUCTION_DIM=hidden_dim (default).
// EXPECTED = num_attention_heads
template <typename Config, typename Globals>
struct o_proj : MatVecAddOp<
    Globals::num_attention_heads,
    &Globals::o_weights, &Globals::attn_out,
    &Globals::hidden_states, OPCODE_O_ProjResidual,
    OPCODE_O_ProjResidual - 1, Config, Globals
> {};
