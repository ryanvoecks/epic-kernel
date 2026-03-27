#pragma once

#include "mixtral.cuh"

using namespace kittens;
using namespace megakernel;

// speculative_expert_predict (opcode 8):
//   Lightweight instruction that predicts top-2 expert indices using the
//   pre-attention residual (hidden_states). Runs concurrently with attention.
//
//   Depends on: previous layer's ExpertDownProjFused (same as QKV)
//   Output: predicted_expert_indices (read speculatively by RmsRouterUpgate loader)
//
//   No matvec pipeline — all computation in consumer warps using direct gmem access.

template <typename Config, typename Globals> struct speculative_expert_predict {
    static constexpr int opcode = OPCODE_SpeculativeExpertPredict;

    struct parsed_instruction {
        int layer_idx;
        __device__ inline parsed_instruction(
            typename Config::instruction_t &instr) {
            layer_idx = instr[1];
        }
        __device__ inline parsed_instruction(megakernel::state<Config> &s)
            : parsed_instruction(s.instruction()) {}
    };

    struct controller {
        static __device__ int
        release_lid(const Globals &g,
                    typename Config::instruction_t &instruction, int &query) {
            // Release all pages immediately — this op doesn't use pages
            return query;
        }
        static __device__ int init_semaphores(const Globals &g,
                                              megakernel::state<Config> &s) {
            return 0;
        }
    };

    struct loader {
        static __device__ void run(const Globals &g,
                                   megakernel::state<Config> &s) {
            // Release all pages — this op uses no shared memory pages
            if (kittens::laneid() < Config::NUM_PAGES) {
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
            parsed_instruction inst{s};

            // Wait for previous layer's ExpertDownProjFused (hidden_states ready)
            if (kittens::warpid() == 0 && kittens::laneid() == 0) {
                s.record(megakernel::TEVENT_AT_GMEM_WAIT);
                if (inst.layer_idx > 0) {
                    while (*(volatile int *)&g.Bar[{inst.layer_idx - 1,
                                                    OPCODE_ExpertDownProjFused - 1, 0}] <
                           static_cast<int>(Globals::qkv_expected_arrivals)) {
                        __nanosleep(Config::GMEM_SPIN_LOOP_SLEEP_NANOS);
                    }
                }
                s.record(megakernel::TEVENT_DONE_GMEM_WAIT);
            }
            kittens::group<Config::NUM_CONSUMER_WARPS>::sync(3);

            // Each warp loads its chunk of hidden_states and ffn_ln_weights
            constexpr int CHUNK = Globals::hidden_dim / Config::NUM_CONSUMER_WARPS;
            int chunk_start = kittens::warpid() * CHUNK;
            int lane = kittens::laneid();

            // Load hidden_states and ffn_ln_weights into registers
            float act[CHUNK / 32];  // Elements per lane
            float nw[CHUNK / 32];
            int ffn_ln_offset = inst.layer_idx * Globals::hidden_dim;

            for (int i = 0; i < CHUNK / 32; i++) {
                int idx = chunk_start + lane + i * 32;
                act[i] = float(g.hidden_states.raw_ptr[idx]);
                nw[i] = float(g.ffn_ln_weights.raw_ptr[ffn_ln_offset + idx]);
            }

            // RMS norm: compute sum of squares
            float partial_sq = 0.f;
            for (int i = 0; i < CHUNK / 32; i++) {
                partial_sq += act[i] * act[i];
            }
            // Reduce within warp
            for (int mask = 16; mask >= 1; mask >>= 1)
                partial_sq += __shfl_xor_sync(0xffffffff, partial_sq, mask);

            // Cross-warp reduction via scratch
            float *scratch = (float *)s.scratch();
            if (kittens::laneid() == 0) {
                scratch[kittens::warpid()] = partial_sq;
            }
            kittens::group<Config::NUM_CONSUMER_WARPS>::sync(0);

            float full_sum = 0.f;
            for (int i = 0; i < Config::NUM_CONSUMER_WARPS; i++) {
                full_sum += scratch[i];
            }
            float rms_scale = rsqrtf(full_sum / static_cast<float>(Globals::hidden_dim) + g.rms_norm_eps);

            // Apply RMS norm: normed = act * rms_scale * weight
            float normed[CHUNK / 32];
            for (int i = 0; i < CHUNK / 32; i++) {
                normed[i] = act[i] * rms_scale * nw[i];
            }

            s.record(megakernel::TEVENT_FIRST_USE);

            // Compute dot products with router_weights for all experts
            float partial_dots[Globals::num_experts];
            for (int e = 0; e < Globals::num_experts; e++) {
                float dot = 0.f;
                int row_base = (inst.layer_idx * Globals::num_experts + e) * Globals::hidden_dim;
                for (int i = 0; i < CHUNK / 32; i++) {
                    int idx = chunk_start + lane + i * 32;
                    float wv = float(g.router_weights.raw_ptr[row_base + idx]);
                    dot += normed[i] * wv;
                }
                // Reduce within warp
                for (int mask = 16; mask >= 1; mask >>= 1)
                    dot += __shfl_xor_sync(0xffffffff, dot, mask);
                partial_dots[e] = dot;
            }

            // Write per-warp partials to scratch
            if (kittens::laneid() == 0) {
                for (int e = 0; e < Globals::num_experts; e++) {
                    scratch[e * Config::NUM_CONSUMER_WARPS + kittens::warpid()] = partial_dots[e];
                }
            }
            kittens::group<Config::NUM_CONSUMER_WARPS>::sync(2);

            // Warp 0, lane 0: reduce across warps, softmax, top-2
            if (kittens::warpid() == 0 && kittens::laneid() == 0) {
                float logits[Globals::num_experts];
                for (int e = 0; e < Globals::num_experts; e++) {
                    float sum = 0.f;
                    for (int w = 0; w < Config::NUM_CONSUMER_WARPS; w++) {
                        sum += scratch[e * Config::NUM_CONSUMER_WARPS + w];
                    }
                    logits[e] = sum;
                }

                // Stable softmax
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

                // Top-2 selection
                int idx0 = 0, idx1 = 1;
                if (probs[idx1] > probs[idx0]) { int tmp = idx0; idx0 = idx1; idx1 = tmp; }
                for (int e = 2; e < Globals::num_experts; e++) {
                    if (probs[e] > probs[idx0]) { idx1 = idx0; idx0 = e; }
                    else if (probs[e] > probs[idx1]) { idx1 = e; }
                }

                // Write predicted indices to global memory
                g.predicted_expert_indices.raw_ptr[0] = idx0;
                g.predicted_expert_indices.raw_ptr[1] = idx1;

                s.record(megakernel::TEVENT_LAST_USE);

                // Signal barrier
                atomicAdd(&g.Bar[{inst.layer_idx, opcode - 1, 0}], 1);
            }
        }
    };

    struct storer {
        static __device__ void run(const Globals &g,
                                   megakernel::state<Config> &s) {
            // No-op — consumer writes directly to gmem
        }
    };
};
