#pragma once

#include "mixtral.cuh"
#include "utils.cuh"
#include "matvec_pipeline.cuh"

using namespace kittens;
using namespace megakernel;

// moe_router (opcode 5):
//   1. Waits for all o_proj blocks to finish (Bar[layer, OPCODE_OProj-1, 0] >= num_attn_heads)
//   2. Computes RMS norm of hidden_states using ffn_ln_weights -> writes router_normed_hidden
//   3. Computes 8 dot products (router_weights @ normed_hidden)
//   4. Softmax + top-2 + renormalize
//   5. Writes selected_expert_indices and selected_expert_scores
//   6. Signals Bar[layer, OPCODE_MoE_Router-1, 0] = 1

template <typename Config, typename Globals> struct moe_router {
    static constexpr int opcode = OPCODE_MoE_Router;
    static constexpr int EXPECTED_OPROJ_ARRIVALS = Globals::num_attention_heads;

    // Semaphore 0: activation + norm weights loaded
    __device__ static inline kittens::semaphore &
    data_loaded(megakernel::state<Config> &s) {
        return s.semaphores()[0];
    }

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
            // Page 0 = activation page (needed); pages 1..12 = released immediately
            return query;
        }
        static __device__ int init_semaphores(const Globals &g,
                                              megakernel::state<Config> &s) {
            init_semaphore(data_loaded(s), 1);
            return 1;
        }
    };

    struct loader {
        static __device__ void run(const Globals &g,
                                   megakernel::state<Config> &s) {
            parsed_instruction inst{s};

            if (kittens::laneid() == 0) {
                // Page 0: first half = hidden_states, second half = ffn_ln_weights
                int pid = s.pid(0);
                s.wait_page_ready(pid);

                // Layout: [sv_bf<hidden_dim>][sv_bf<hidden_dim>]
                auto &norm_smem = *reinterpret_cast<kittens::sv_bf<Globals::hidden_dim> *>(
                    s.pages[pid].ptr(sizeof(kittens::sv_bf<Globals::hidden_dim>)));

                auto &sem = data_loaded(s);
                kittens::tma::expect(sem, norm_smem);
                kittens::tma::load_async<cache_policy::EVICT_LAST>(
                    norm_smem, g.ffn_ln_weights, {inst.layer_idx, 0}, sem);
            }

            // Release pages 1..NUM_PAGES-1 immediately
            if (kittens::laneid() >= 1 && kittens::laneid() < Config::NUM_PAGES) {
                int pid = s.pid(kittens::laneid());
                s.wait_page_ready(pid);
                s.finish_page(pid, Config::NUM_CONSUMER_WARPS);
            }
        }
    };

    struct launcher {
        static __device__ void run(const Globals &g,
                                   megakernel::state<Config> &s) {
#ifdef KITTENS_BLACKWELL
            if (kittens::laneid() == 0) {
                s.wait_tensor_ready();
                arrive(s.tensor_finished, Config::NUM_CONSUMER_WARPS);
            }
#endif
        }
    };

    struct consumer {
        static __device__ void run(const Globals &g,
                                   megakernel::state<Config> &s) {
            parsed_instruction inst{s};

            // All warps participate in barrier wait and RMS norm
            // Warp 0, lane 0 spins on the o_proj barrier
            if (kittens::warpid() == 0 && kittens::laneid() == 0) {
                s.record(megakernel::TEVENT_AT_GMEM_WAIT);
                while (*(volatile int *)&g.Bar[{inst.layer_idx, OPCODE_OProj - 1, 0}] <
                       EXPECTED_OPROJ_ARRIVALS) {
                    __nanosleep(Config::GMEM_SPIN_LOOP_SLEEP_NANOS);
                }
                s.record(megakernel::TEVENT_DONE_GMEM_WAIT);
            }
            kittens::group<Config::NUM_CONSUMER_WARPS>::sync(0);

            // Wait for norm weights to be loaded via TMA
            kittens::wait(data_loaded(s), 0);

            // Get activation page pointers
            int pid = s.pid(0);
            constexpr int CHUNK = Globals::hidden_dim / Config::NUM_CONSUMER_WARPS;
            using sv_chunk = kittens::sv_bf<CHUNK>;
            using rv_chunk = kittens::rv_fl<CHUNK>;

            // Each warp's activation chunk in page 0 first half
            sv_chunk &act_smem = reinterpret_cast<sv_chunk *>(
                s.pages[pid].ptr())[kittens::warpid()];
            // Each warp's norm-weight chunk in page 0 second half
            sv_chunk &nw_smem = reinterpret_cast<sv_chunk *>(
                s.pages[pid].ptr(sizeof(kittens::sv_bf<Globals::hidden_dim>)))[kittens::warpid()];

            // Load hidden_states chunk from GMEM into SMEM
            // coord<>{} uses default_type (no tile-size scaling) so element offset = warpid*CHUNK
            kittens::warp::load(act_smem, g.hidden_states,
                                kittens::coord<>{kittens::warpid() * CHUNK});
            kittens::warp::sync();

            // RMS norm across all warps
            rv_chunk act_vec = rms_norm<Config, Globals>(
                nw_smem, act_smem, g.rms_norm_eps,
                s.scratch());
            kittens::warp::sync();

            // Store normed chunk to router_normed_hidden GMEM
            // Convert rv_fl<CHUNK> to bf16 and write
            {
                // Reuse act_smem as bf16 temp buffer (already consumed above)
                sv_chunk &tmp_smem = act_smem;
                kittens::warp::store(tmp_smem, act_vec);
                kittens::warp::sync();
                // coord<>{} uses default_type (no tile-size scaling)
                kittens::warp::store(g.router_normed_hidden, tmp_smem,
                                     kittens::coord<>{kittens::warpid() * CHUNK});
            }

            // All-warp barrier: ensures every warp's GMEM write to router_normed_hidden
            // is globally visible before warp 0 reads it for the dot-product loop.
            // Uses barrier id=1 (rms_norm above uses id=0).
            kittens::group<Config::NUM_CONSUMER_WARPS>::sync(1);

            // Release activation page
            s.warp_finish_page(pid, 1);

            // Only warp 0 computes the 8 dot products
            if (kittens::warpid() != 0) return;

            float logits[Globals::num_experts];
            int lane = kittens::laneid();

            // Compute 8 dot products: router_weights[layer*ne+e, :] . normed_hidden
            for (int e = 0; e < Globals::num_experts; e++) {
                float dot = 0.f;
                int row_base = (inst.layer_idx * Globals::num_experts + e) *
                               Globals::hidden_dim;
                for (int i = lane; i < Globals::hidden_dim; i += 32) {
                    float nv = float(g.router_normed_hidden.raw_ptr[i]);
                    float wv = float(g.router_weights.raw_ptr[row_base + i]);
                    dot += nv * wv;
                }
                // Warp-level reduction
                for (int mask = 16; mask >= 1; mask >>= 1) {
                    dot += __shfl_xor_sync(0xffffffff, dot, mask);
                }
                logits[e] = dot;
            }

            // Softmax + top-2 (only lane 0)
            if (lane == 0) {
                float max_val = logits[0];
                for (int e = 1; e < Globals::num_experts; e++) {
                    max_val = fmaxf(max_val, logits[e]);
                }
                float sum_exp = 0.f;
                float probs[Globals::num_experts];
                for (int e = 0; e < Globals::num_experts; e++) {
                    probs[e] = __expf(logits[e] - max_val);
                    sum_exp += probs[e];
                }
                for (int e = 0; e < Globals::num_experts; e++) {
                    probs[e] /= sum_exp;
                }

                // Top-2 selection
                int idx0 = 0, idx1 = 1;
                if (probs[idx1] > probs[idx0]) {
                    int tmp = idx0; idx0 = idx1; idx1 = tmp;
                }
                for (int e = 2; e < Globals::num_experts; e++) {
                    if (probs[e] > probs[idx0]) {
                        idx1 = idx0; idx0 = e;
                    } else if (probs[e] > probs[idx1]) {
                        idx1 = e;
                    }
                }

                // Renormalize top-2
                float top2_sum = probs[idx0] + probs[idx1];
                float score0 = probs[idx0] / top2_sum;
                float score1 = probs[idx1] / top2_sum;

                // Write outputs
                g.selected_expert_indices.raw_ptr[0] = idx0;
                g.selected_expert_indices.raw_ptr[1] = idx1;
                g.selected_expert_scores.raw_ptr[0] = __float2bfloat16(score0);
                g.selected_expert_scores.raw_ptr[1] = __float2bfloat16(score1);
            }
            kittens::warp::sync();
        }
    };

    struct storer {
        static __device__ void run(const Globals &g,
                                   megakernel::state<Config> &s) {
            if (kittens::warpid() == 0 && kittens::laneid() == 0) {
                parsed_instruction inst{s};
                asm volatile("fence.acq_rel.gpu;\n" ::: "memory");
                s.record(megakernel::TEVENT_AT_GMEM_STORE);
                atomicExch(&g.Bar[{inst.layer_idx, opcode - 1, 0}], 1);
                s.record(megakernel::TEVENT_DONE_GMEM_STORE);
            }
        }
    };
};
