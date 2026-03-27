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

    struct parsed_instruction {
        int layer_idx, start_block_idx, num_blocks, iters;
        __device__ inline parsed_instruction(
            typename Config::instruction_t &instr) {
            layer_idx       = instr[1];
            start_block_idx = instr[2];
            num_blocks      = instr[3];
            // gate + up interleaved x 2 experts
            iters           = 2 * num_blocks * Globals::num_experts_per_tok;
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
            // iter layout: for each expert_loop in [0, num_experts_per_tok):
            //   for each block in [0, num_blocks):
            //     even sub-iter: up weights
            //     odd sub-iter: gate weights
            // Total iters = 2 * num_blocks * num_experts_per_tok
            int expert_loop = iter / (2 * inst.num_blocks);
            int within_expert = iter % (2 * inst.num_blocks);
            int block_offset = within_expert / 2;
            int block_idx = inst.start_block_idx + block_offset;

            // Get selected expert index at runtime
            int expert_idx = g.selected_expert_indices.raw_ptr[expert_loop];
            int depth = inst.layer_idx * Globals::num_experts + expert_idx;

            if (within_expert % 2 == 0) {
                // even: up weights
                kittens::tma::load_async<dim::ROW, cache_policy::EVICT_FIRST>(
                    weight_chunk, g.expert_up_weights,
                    {0, depth, block_idx, col_idx}, sem);
            } else {
                // odd: gate weights
                kittens::tma::load_async<dim::ROW, cache_policy::EVICT_FIRST>(
                    weight_chunk, g.expert_gate_weights,
                    {0, depth, block_idx, col_idx}, sem);
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
            int expert_idx = g.selected_expert_indices.raw_ptr[expert_loop];

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
                                     pipeline_specifics>;

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

                // Wait for the consumer to finish computing the router and
                // writing selected_expert_indices/scores. The pipeline's
                // load_iter reads these indices to determine which expert's
                // weights to load, so we must not start loading until they
                // are written.
                kittens::wait(router_done(s), 0);
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

            // Release activation page and reload normed activations for pipeline
            s.warp_finish_page(activation_page, 1);

            sv_t &act_smem_pipe = reinterpret_cast<sv_t *>(
                &pipeline::get_activations(s))[kittens::warpid()];

            kittens::warp::load(act_smem_pipe, g.router_normed_hidden,
                                kittens::coord<>{kittens::warpid() * pipeline::REDUCTION_DIM_PER_WARP});
            kittens::warp::sync();

            rv_t act_vec_pipe;
            kittens::warp::load(act_vec_pipe, act_smem_pipe);
            kittens::warp::sync();

            // ---- Phase 2: UpGate pipeline ----
            pipeline::consumer_loop(s, g, act_vec_pipe);
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

                // Signal barrier: all upgate blocks done for both experts
                atomicAdd(&g.Bar[{inst.layer_idx, opcode - 1, 0}],
                          inst.num_blocks);
                s.record(megakernel::TEVENT_DONE_GMEM_STORE);
            }
        }
    };
};
