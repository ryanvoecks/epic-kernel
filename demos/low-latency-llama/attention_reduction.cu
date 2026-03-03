// Attention reduction instruction for combining partial reduction outputs

#include <limits>

#include "llama.cuh"

using namespace kittens;
using namespace megakernel;

using globals = llama_1b_globals;

constexpr uint32_t Q_HEADS_PER_INSTRUCTION = 4;
constexpr uint32_t MAX_ATTN_PARTIALS = globals::sm_count;
constexpr uint32_t ROUNDED_MAX_ATTN_PARTIALS = ((MAX_ATTN_PARTIALS + 15) / 16) * 16;

using l_partial_sv = kittens::sv_fl<ROUNDED_MAX_ATTN_PARTIALS>;
using o_sv = kittens::sv_fl<globals::head_dim>;
using o_rv = kittens::rv_fl<globals::head_dim>;
using o_final_sv = kittens::sv_bf<globals::head_dim>;

template <typename Config, typename Globals> struct attention_reduction {
    // Opcode
    static constexpr uint32_t opcode = OPCODE_AttentionReduction;
    static constexpr uint32_t prev_opcode = OPCODE_PartialAttention;

    // SMEM allocation
    static constexpr uint32_t SMEM_PER_HEAD = sizeof(l_partial_sv) + sizeof(o_final_sv);
    static constexpr uint32_t SMEM_PER_STAGE = Q_HEADS_PER_INSTRUCTION * sizeof(o_sv);
    static constexpr uint32_t SMEM_FIXED = Q_HEADS_PER_INSTRUCTION * SMEM_PER_HEAD;
    static constexpr uint32_t SMEM_MAX_STAGES = (Config::PAGE_SIZE - SMEM_FIXED) / SMEM_PER_STAGE;

    // Semaphores
    static constexpr uint32_t SEMS_PER_HEAD = 3;    // L_arrived + L_finished + final_O_ready
    static constexpr uint32_t SEMS_PER_STAGE = 2;   // O_partial_arrived + O_partial_finished
    static constexpr uint32_t SEMS_FIXED = Q_HEADS_PER_INSTRUCTION * SEMS_PER_HEAD;
    static constexpr uint32_t SEMS_FREE = Config::DYNAMIC_SEMAPHORES - SEMS_FIXED;
    static constexpr uint32_t SEMS_MAX_STAGES = SEMS_FREE / SEMS_PER_STAGE;

    // Max number of stages determined by SMEM page size and number of semaphores
    static constexpr uint32_t NUM_STAGES =
        std::min({SMEM_MAX_STAGES, SEMS_MAX_STAGES, MAX_ATTN_PARTIALS});

    struct parsed_instruction {
        int layer_idx;
        int q_head_start_idx;
        int num_partials;
        int is_terminal;
        int reduction_list_length;
        int reduction_list[ROUNDED_MAX_ATTN_PARTIALS];

        __device__ inline parsed_instruction(megakernel::state<Config> &s) {
            layer_idx = s.instruction()[1];
            q_head_start_idx = s.instruction()[2];
            num_partials = s.instruction()[3];
            is_terminal = s.instruction()[4]; // not used
            reduction_list_length = s.instruction()[5];

#pragma unroll
            for (int k = 0; k < MAX_ATTN_PARTIALS; ++k) {
                if (k < reduction_list_length) {
                    reduction_list[k] = s.instruction()[6 + k];
                }
            }
        }
    };

    // --- kittens::semaphore Access Helpers ---
    __device__ static constexpr int
    O_partial_sem_idx(int stage, bool is_finished) {
        return stage * SEMS_PER_STAGE + static_cast<uint32_t>(is_finished);
    }
    __device__ static constexpr int L_partial_sem_idx(int q_head_local_idx,
                                                      bool is_finished) {
        return NUM_STAGES * SEMS_PER_STAGE +
            q_head_local_idx * SEMS_PER_HEAD +
            static_cast<uint32_t>(is_finished);
    }
    __device__ static constexpr int
    Final_O_ready_sem_idx(int q_head_local_idx) {
        return NUM_STAGES * SEMS_PER_STAGE + q_head_local_idx * SEMS_PER_HEAD + 2;
    }

    __device__ static inline kittens::semaphore &
    O_partial_arrived(megakernel::state<config> &s, int stage) {
        return s.semaphores()[O_partial_sem_idx(stage, false)];
    }
    __device__ static inline kittens::semaphore &
    O_partial_finished(megakernel::state<config> &s, int stage) {
        return s.semaphores()[O_partial_sem_idx(stage, true)];
    }
    __device__ static inline kittens::semaphore &
    L_partial_all_arrived(megakernel::state<config> &s, int q_head_local_idx) {
        return s.semaphores()[L_partial_sem_idx(q_head_local_idx, false)];
    }
    __device__ static inline kittens::semaphore &
    L_partial_all_finished(megakernel::state<config> &s, int q_head_local_idx) {
        return s.semaphores()[L_partial_sem_idx(q_head_local_idx, true)];
    }
    __device__ static inline kittens::semaphore &final_O_ready(megakernel::state<config> &s,
                                                      int q_head_local_idx) {
        return s.semaphores()[Final_O_ready_sem_idx(q_head_local_idx)];
    }

    // --- Shared Memory Page Management Helpers ---
    static constexpr int SHARED_DATA_PAGE =
        0; // Use only the first logical page

    __device__ static inline void wait_shared_page(megakernel::state<Config> &s) {
        if (kittens::warp::laneid() == 0) {
            s.wait_page_ready(s.pid(SHARED_DATA_PAGE));
        }
    }
    __device__ static inline void finish_shared_page(megakernel::state<Config> &s) {
        if (kittens::warp::laneid() == 0) {
            s.finish_page(s.pid(SHARED_DATA_PAGE), Config::NUM_CONSUMER_WARPS);
        }
    }

    // --- Shared Memory Layout and Access Helpers (Single Page) ---
    static constexpr size_t size_per_head = SMEM_PER_HEAD + NUM_STAGES * sizeof(o_sv);
    static constexpr size_t total_smem_needed = SMEM_FIXED + NUM_STAGES * SMEM_PER_STAGE;
    static_assert(total_smem_needed <= config::PAGE_SIZE,
                  "Required shared memory exceeds configured page size.");

    __device__ static inline l_partial_sv &
    get_L_partial_smem(megakernel::state<config> &s, int q_head_local_idx) {
        int pid = s.pid(SHARED_DATA_PAGE);
        char *page_base_ptr = reinterpret_cast<char *>(s.pages[pid].data);
        char *head_base_ptr = page_base_ptr + q_head_local_idx * size_per_head;
        return *reinterpret_cast<l_partial_sv *>(head_base_ptr);
    }
    __device__ static inline o_sv &
    get_O_partial_smem(megakernel::state<config> &s, int q_head_local_idx, int stage) {
        int pid = s.pid(SHARED_DATA_PAGE);
        char *page_base_ptr = reinterpret_cast<char *>(s.pages[pid].data);
        char *head_base_ptr = page_base_ptr + q_head_local_idx * size_per_head;
        size_t offset = sizeof(l_partial_sv) + stage * sizeof(o_sv);
        return *reinterpret_cast<o_sv *>(head_base_ptr + offset);
    }
    __device__ static inline o_final_sv &
    get_O_final_smem(megakernel::state<config> &s, int q_head_local_idx) {
        int pid = s.pid(SHARED_DATA_PAGE);
        char *page_base_ptr = reinterpret_cast<char *>(s.pages[pid].data);
        char *head_base_ptr = page_base_ptr + q_head_local_idx * size_per_head;
        size_t offset = sizeof(l_partial_sv) + NUM_STAGES * sizeof(o_sv);
        return *reinterpret_cast<o_final_sv *>(head_base_ptr + offset);
    }

    struct controller {
        static __device__ int
        release_lid(const Globals &g,
                    typename Config::instruction_t &instruction, int &query) {
            int ret_order[13] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0};
            return ret_order[query];
        }
        static __device__ int init_semaphores(const Globals &g,
                                              megakernel::state<Config> &s) {
            for (int stage = 0; stage < NUM_STAGES; stage++) {
                init_semaphore(O_partial_arrived(s, stage), 0, Q_HEADS_PER_INSTRUCTION);
                init_semaphore(O_partial_finished(s, stage), 0, Q_HEADS_PER_INSTRUCTION);
            }
            for (int q_head = 0; q_head < Q_HEADS_PER_INSTRUCTION; ++q_head) {
                init_semaphore(L_partial_all_arrived(s, q_head), 0, 1);
                init_semaphore(L_partial_all_finished(s, q_head), 0, 1);
                init_semaphore(final_O_ready(s, q_head), 0, 1);
            }
            return NUM_STAGES * SEMS_PER_STAGE + SEMS_FIXED;
        }
    };

    struct loader {
        static __device__ void run(const Globals &g, megakernel::state<Config> &s) {
            auto laneid = kittens::warp::laneid();

            if (laneid == 0) {
                wait_shared_page(s);
            } else if (laneid < Config::NUM_PAGES) {
                s.wait_page_ready(s.pid(laneid));
                s.finish_page(s.pid(laneid), Config::NUM_CONSUMER_WARPS);
            }
            kittens::warp::sync(); // Have to make sure lane 0 finished waiting
        }
    };

    struct launcher {
        static __device__ void run(const Globals &g, megakernel::state<Config> &s) {
            if (kittens::warp::laneid() == 0) {
                // Immediately release tensor engine for next instruction (unused)
                s.wait_tensor_ready();
                arrive(s.tensor_finished, Config::NUM_CONSUMER_WARPS);

                // Instruction details
                const parsed_instruction inst{s};

                // Wait until all partial attention outputs are ready in global memory
                const int32_t kv_head_idx = inst.q_head_start_idx / Q_HEADS_PER_INSTRUCTION;
                const kittens::coord<> barrier_idx = {
                    inst.layer_idx, prev_opcode - 1, kv_head_idx
                };
                s.record(megakernel::TEVENT_AT_GMEM_WAIT);
                while (megakernel::gmem_read(&g.Bar[barrier_idx]) < inst.num_partials) {
                    __nanosleep(Config::GMEM_SPIN_LOOP_SLEEP_NANOS);
                }
                s.record(megakernel::TEVENT_DONE_GMEM_WAIT);

                // Async load of log-sum-exponent intermediates from Q heads into SMEM
                for (int i = 0; i < Q_HEADS_PER_INSTRUCTION; i++) {
                    l_partial_sv& L_smem = get_L_partial_smem(s, i);
                    const kittens::coord<> q_head_idx = {0, 0, inst.q_head_start_idx + i, 0};
                    kittens::tma::expect(L_partial_all_arrived(s, i), L_smem);
                    kittens::tma::load_async<cache_policy::EVICT_FIRST>(
                        L_smem, g.attn_lse_intermediates, q_head_idx, L_partial_all_arrived(s, i));
                }

                // Load output intermediates from each partial attention contributor
                // All Q heads for a given partial share one per-stage semaphore
                for (int i = 0; i < inst.num_partials; ++i) {
                    const uint8_t stage = i % NUM_STAGES;
                    const uint8_t cur_partial_idx = inst.reduction_list[i];

                    // Wait until SMEM stage is free (all consumer warps finished with it)
                    if (i >= NUM_STAGES) {
                        const uint8_t prev_phase = (i / NUM_STAGES - 1) % 2;
                        kittens::wait(O_partial_finished(s, stage), prev_phase);
                    }

                    // Issue TMA loads for all Q heads under the shared per-stage semaphore
                    for (int j = 0; j < Q_HEADS_PER_INSTRUCTION; ++j) {
                        o_sv& O_smem = get_O_partial_smem(s, j, stage);
                        const kittens::coord<> partial_o_idx = {
                            0, inst.q_head_start_idx + j, cur_partial_idx, 0
                        };
                        kittens::tma::expect(O_partial_arrived(s, stage), O_smem);
                        kittens::tma::load_async<cache_policy::EVICT_FIRST>(
                            O_smem,
                            g.attn_out_intermediates,
                            partial_o_idx,
                            O_partial_arrived(s, stage)
                        );
                    }
                }
            }
        }
    };

    struct consumer {
        static __device__ void run(const Globals &g, megakernel::state<Config> &s) {

            if (kittens::warpid() < Q_HEADS_PER_INSTRUCTION) {

                parsed_instruction inst{s};
                int q_head_local_idx = kittens::warpid();

                o_rv accumulated_out;
                float accumulated_lse = -INFINITY;

                o_rv current_out;
                float current_lse;

                kittens::warp::zero(accumulated_out);

                kittens::warp::wait(L_partial_all_arrived(s, q_head_local_idx), 0);
                if (kittens::laneid() == 0)
                    s.record(megakernel::TEVENT_CONSUMER_START + 16 + kittens::warpid());
                l_partial_sv &L_smem = get_L_partial_smem(s, q_head_local_idx);

                // --- Reduction Pipeline ---
                for (int i = 0; i < inst.num_partials; ++i) {
                    int stage = i % NUM_STAGES;
                    kittens::warp::wait(O_partial_arrived(s, stage),
                               (i / NUM_STAGES) % 2);

                    o_sv &O_smem =
                        get_O_partial_smem(s, q_head_local_idx, stage);

                    // Load cur L_partial value
                    int cur_partial_idx = inst.reduction_list[i];
                    uint32_t src_ptr_L =
                        static_cast<uint32_t>(__cvta_generic_to_shared(
                            &L_smem.data[cur_partial_idx]));
                    move<float>::lds(current_lse, src_ptr_L);
                    // Load O_partial_reg
                    kittens::warp::load(current_out, O_smem);

                    float max_lse = max(accumulated_lse, current_lse);

                    float accumulated_exp = exp2f(accumulated_lse - max_lse);
                    float current_exp = exp2f(current_lse - max_lse);

                    float new_denom = accumulated_exp + current_exp;

                    float accumulated_scale = accumulated_exp / new_denom;
                    float current_scale = current_exp / new_denom;

                    kittens::warp::mul(accumulated_out, accumulated_out,
                              accumulated_scale);
                    kittens::warp::mul(current_out, current_out, current_scale);
                    kittens::warp::add(accumulated_out, accumulated_out, current_out);

                    // Update LSE accumulator:
                    accumulated_lse = max_lse + log2f(new_denom);

                    kittens::warp::arrive(O_partial_finished(s, stage));
                }
                kittens::warp::arrive(L_partial_all_finished(s, q_head_local_idx));

                o_final_sv &O_final_smem =
                    get_O_final_smem(s, q_head_local_idx);
                kittens::warp::store(O_final_smem, accumulated_out);
                kittens::warp::sync();

                kittens::warp::arrive(final_O_ready(s, q_head_local_idx));
            }
        }
    };

    // Storer kittens::warp: Responsible for storing data from shared memory back to
    // global memory.
    struct storer {
        static __device__ void run(const Globals &g, megakernel::state<Config> &s) {
            parsed_instruction inst{s};
            if (kittens::warp::laneid() < Q_HEADS_PER_INSTRUCTION) {
                int q_head_local_idx = kittens::warp::laneid();

                o_final_sv &O_final_smem =
                    get_O_final_smem(s, q_head_local_idx);
                kittens::wait(final_O_ready(s, q_head_local_idx), 0);
                if (kittens::warp::laneid() == 0) {
                    s.record(megakernel::TEVENT_OUTPUT_READY);
                }

                kittens::tma::store_async<cache_policy::NORMAL>(
                    g.attn_out, O_final_smem,
                    {0, 0, 0, inst.q_head_start_idx + q_head_local_idx});
                kittens::tma::store_async_wait();

                // atomicAdd(&g.Bar[{inst.layer_idx, opcode - 1,
                // inst.q_head_start_idx + q_head_local_idx}], 1);
            }
            finish_shared_page(s);

            kittens::warp::sync();
            if (kittens::warp::laneid() == 0) {
                s.record(megakernel::TEVENT_AT_GMEM_STORE);
                // asm volatile("fence.acq_rel.gpu;");

                // simple signalling strat for now
                atomicAdd(&g.Bar[{inst.layer_idx, opcode - 1, 0}],
                          Q_HEADS_PER_INSTRUCTION);
                s.record(megakernel::TEVENT_DONE_GMEM_STORE);
            }
        }
    };
};
