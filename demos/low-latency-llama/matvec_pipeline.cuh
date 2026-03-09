#pragma once

#include "llama.cuh"

// REDUCTION_DIM: number of columns reduced per pipeline iteration.
//   Default = Globals::hidden_dim (3072 for 3B).
//   For downproj: 1024 (matvec_reduction_size, since 8192/3072 is not integer).
//
// TILE_WIDTH and STAGE_PAGES are computed so that:
//   1. STAGE_PAGES divides NUM_CONSUMER_WARPS (warp-to-page mapping)
//   2. Each tile (16 x TILE_WIDTH x 2 bytes) fits in one page
//   3. At least 1 pipeline stage fits in the page budget
//
//   hidden_dim=3072 -> TILE_WIDTH=384, STAGE_PAGES=8, INPUT_PIPELINE_STAGES=1
//   reduction=1024  -> TILE_WIDTH=512, STAGE_PAGES=2, INPUT_PIPELINE_STAGES=6
template <typename Config, typename Globals, typename parsed_instruction,
          typename pipeline_specifics,
          int REDUCTION_DIM = Globals::hidden_dim>
struct matvec_pipeline {
    // Find smallest STAGE_PAGES (SP) that divides NUM_CONSUMER_WARPS,
    // divides REDUCTION_DIM, and whose tile width fits in a page.
    static constexpr int find_stage_pages() {
        const int NCW = Config::NUM_CONSUMER_WARPS;
        const int MAX_TILE_W = Config::PAGE_SIZE / (16 * 2); // 512
        for (int sp = 1; sp <= NCW; sp++) {
            if (NCW % sp != 0) continue;
            if (REDUCTION_DIM % sp != 0) continue;
            int tw = REDUCTION_DIM / sp;
            if (tw <= MAX_TILE_W && 1 + sp <= Config::NUM_PAGES)
                return sp;
        }
        return -1;
    }
    static constexpr int STAGE_PAGES = find_stage_pages();
    static constexpr int TILE_WIDTH = REDUCTION_DIM / STAGE_PAGES;
    static constexpr int INPUT_PIPELINE_STAGES =
        (Config::NUM_PAGES - 1) / STAGE_PAGES;
    static constexpr int OUTPUT_PIPELINE_STAGES = 3;
    static constexpr int ACTIVATION_PAGE = 0;
    static constexpr int WEIGHTS_START_PAGE = 1;
    static constexpr int TOTAL_WEIGHT_PAGES =
        INPUT_PIPELINE_STAGES * STAGE_PAGES;

    static_assert(STAGE_PAGES > 0,
                  "No valid STAGE_PAGES found for REDUCTION_DIM");
    static_assert(TILE_WIDTH % 16 == 0,
                  "TILE_WIDTH must be a multiple of 16 for TK alignment");
    static_assert(1 + TOTAL_WEIGHT_PAGES <= Config::NUM_PAGES,
                  "Not enough pages for activation + weight pipeline stages");

    static constexpr int REDUCTION_DIM_PER_WARP =
        REDUCTION_DIM / Config::NUM_CONSUMER_WARPS;

    static constexpr int SEM_COUNT =
        1 + (INPUT_PIPELINE_STAGES + OUTPUT_PIPELINE_STAGES) * 2;

    static constexpr int SCRATCH_BYTES_PER_WARP = 16 * sizeof(float);
    static constexpr int SCRATCH_BYTES_PER_STAGE =
        SCRATCH_BYTES_PER_WARP * Config::NUM_CONSUMER_WARPS;
    static constexpr int USED_SCRATCH_BYTES =
        OUTPUT_PIPELINE_STAGES * SCRATCH_BYTES_PER_STAGE;
    static_assert(USED_SCRATCH_BYTES <= Config::SCRATCH_BYTES,
                  "USED_SCRATCH_BYTES must be less than SCRATCH_BYTES");

    // Pages
    __device__ static inline int get_activation_page(megakernel::state<Config> &s) {
        return s.pid(ACTIVATION_PAGE);
    }

    __device__ static inline int get_weight_page(megakernel::state<Config> &s,
                                                 int stage, int offset) {
        return s.pid(WEIGHTS_START_PAGE + stage * STAGE_PAGES + offset);
    }

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

    __device__ static inline kittens::sv_bf<REDUCTION_DIM> &
    get_activations(megakernel::state<Config> &s) {
        return *reinterpret_cast<kittens::sv_bf<REDUCTION_DIM> *>(
            s.pages[get_activation_page(s)].ptr());
    }

    __device__ static inline uint8_t *
    get_output_start(megakernel::state<Config> &s, int stage) {
        return (uint8_t *)s.scratch() + (stage * SCRATCH_BYTES_PER_STAGE);
    }

    __device__ static inline int
    release_lid(const Globals &g, typename Config::instruction_t &instruction,
                int &query) {
        parsed_instruction inst{instruction};
        auto iters = inst.iters;

        constexpr int TOTAL_ALLOC = 1 + INPUT_PIPELINE_STAGES * STAGE_PAGES;
        constexpr int EXTRA = Config::NUM_PAGES - TOTAL_ALLOC;

        // Release any pages beyond our allocation first
        if (query < EXTRA)
            return TOTAL_ALLOC + query;

        int inner = query - EXTRA;

        // Determine stage usage
        int stages_used =
            (iters < INPUT_PIPELINE_STAGES) ? iters : INPUT_PIPELINE_STAGES;
        int last_stage =
            (iters > 0) ? ((iters - 1) % INPUT_PIPELINE_STAGES) : 0;

        // Build release order: unused_stages -> activation -> used_stages (last
        // stage last)
        int order[TOTAL_ALLOC];
        int idx = 0;

        // Pages from unused pipeline stages (never loaded)
        for (int s = stages_used; s < INPUT_PIPELINE_STAGES; s++) {
            for (int p = 0; p < STAGE_PAGES; p++) {
                order[idx++] = WEIGHTS_START_PAGE + s * STAGE_PAGES + p;
            }
        }

        // Activation page
        order[idx++] = ACTIVATION_PAGE;

        // Used stages, last-used stage goes last
        if (stages_used > 0) {
            for (int i = 1; i <= stages_used; i++) {
                int s = (last_stage + i) % stages_used;
                for (int p = 0; p < STAGE_PAGES; p++) {
                    order[idx++] = WEIGHTS_START_PAGE + s * STAGE_PAGES + p;
                }
            }
        }

        return order[inner];
    }

    __device__ static inline int init_semaphores(megakernel::state<Config> &s) {
        init_semaphore(activations_arrived(s), 1);
        for (int i = 0; i < INPUT_PIPELINE_STAGES; i++) {
            init_semaphore(weights_arrived(s, i), 1);
            init_semaphore(weights_finished(s, i),
                           Config::NUM_CONSUMER_WARPS);
        }
        for (int i = 0; i < OUTPUT_PIPELINE_STAGES; i++) {
            init_semaphore(outputs_arrived(s, i),
                           Config::NUM_CONSUMER_WARPS);
            init_semaphore(outputs_finished(s, i), 1);
        }
        return SEM_COUNT;
    }

    __device__ static inline void
    loader_loop(megakernel::state<Config> &s, const Globals &g) {
        parsed_instruction inst{s};

        auto needed_pages =
            1 + min(inst.iters, INPUT_PIPELINE_STAGES) * STAGE_PAGES;

        if (kittens::laneid() == 0) {
            int input_stage = 0;
            for (int iter = 0; iter < inst.iters; iter++) {
                kittens::wait(
                    weights_finished(s, input_stage),
                    (iter % (2 * INPUT_PIPELINE_STAGES)) <
                        INPUT_PIPELINE_STAGES);

                auto &sem = weights_arrived(s, input_stage);
                kittens::tma::expect_bytes(
                    sem,
                    sizeof(kittens::bf16) * REDUCTION_DIM * 16);
#pragma unroll
                for (int i = 0; i < STAGE_PAGES; i++) {
                    int weight_page =
                        get_weight_page(s, input_stage, i);
                    if (iter < INPUT_PIPELINE_STAGES) {
                        s.wait_page_ready(weight_page);
                    }
                    auto &weight_chunk =
                        reinterpret_cast<kittens::st_bf<16, TILE_WIDTH> &>(
                            s.pages[weight_page]);

                    if (iter == 0 && i == 0) {
                        s.record(megakernel::TEVENT_FIRST_LOAD);
                    } else if (iter == inst.iters - 1 &&
                               i == STAGE_PAGES - 1) {
                        s.record(megakernel::TEVENT_LAST_LOAD);
                    }

                    pipeline_specifics::load_iter(
                        s, g, inst, iter, i, weight_chunk, sem);
                }

                input_stage = (input_stage + 1) % INPUT_PIPELINE_STAGES;
            }
        } else if (kittens::laneid() >= needed_pages &&
                   kittens::laneid() < Config::NUM_PAGES) {
            auto pid = s.pid(kittens::laneid());
            s.wait_page_ready(pid);
            s.finish_page(pid, Config::NUM_CONSUMER_WARPS);
        }
    }

    template <typename rv_t>
    __device__ static inline void
    consumer_loop(megakernel::state<Config> &s, const Globals &g,
                  rv_t &activations_vec) {
        parsed_instruction inst{s};

        static_assert(
            Config::NUM_CONSUMER_WARPS % STAGE_PAGES == 0,
            "NUM_CONSUMER_WARPS must be divisible by STAGE_PAGES");
        constexpr int WARPS_PER_PAGE =
            Config::NUM_CONSUMER_WARPS / STAGE_PAGES;

        int page_index = kittens::warpid() / WARPS_PER_PAGE;

        int input_stage = 0, output_stage = 0;
        for (int i = 0; i < inst.iters; i++) {
            int weight_page =
                get_weight_page(s, input_stage, page_index);
            kittens::wait(
                weights_arrived(s, input_stage),
                (i % (2 * INPUT_PIPELINE_STAGES)) >=
                    INPUT_PIPELINE_STAGES);
            kittens::wait(
                outputs_finished(s, output_stage),
                (i % (2 * OUTPUT_PIPELINE_STAGES)) <
                    OUTPUT_PIPELINE_STAGES);
            kittens::st_bf<16, REDUCTION_DIM_PER_WARP> &weights =
                reinterpret_cast<
                    kittens::st_bf<16, REDUCTION_DIM_PER_WARP> *>(
                    s.pages[weight_page]
                        .ptr())[kittens::warpid() % WARPS_PER_PAGE];

            kittens::sv_fl<16> &out_smem =
                *reinterpret_cast<kittens::sv_fl<16> *>(
                    get_output_start(s, output_stage) +
                    (kittens::warpid() * SCRATCH_BYTES_PER_WARP));

            if (i == 0) {
                s.record(megakernel::TEVENT_FIRST_USE);
            } else if (i == inst.iters - 1) {
                s.record(megakernel::TEVENT_LAST_USE);
            }

            matvec(out_smem, weights, activations_vec);

            kittens::warp::sync();
            kittens::warp::arrive(outputs_arrived(s, output_stage));
            kittens::warp::arrive(weights_finished(s, input_stage));

            if (i >= inst.iters - INPUT_PIPELINE_STAGES) {
#pragma unroll
                for (int j = 0; j < STAGE_PAGES; j++) {
                    s.warp_finish_page(
                        get_weight_page(s, input_stage, j), 1);
                }
            }

            input_stage = (input_stage + 1) % INPUT_PIPELINE_STAGES;
            output_stage = (output_stage + 1) % OUTPUT_PIPELINE_STAGES;
        }
    }

    template <int iter_scale = 1>
    __device__ static inline void
    storer_loop(megakernel::state<Config> &s, const Globals &g) {
        parsed_instruction inst{s};

        int output_stage = 0;
        for (int i = 0; i < inst.iters; i++) {
            auto &sem = outputs_arrived(s, output_stage);
            auto bit = (i % (2 * OUTPUT_PIPELINE_STAGES)) >=
                       OUTPUT_PIPELINE_STAGES;

            kittens::wait(sem, bit);

            if (i == 0) {
                s.record(megakernel::TEVENT_FIRST_STORE);
            } else if (i == inst.iters - 1) {
                s.record(megakernel::TEVENT_LAST_STORE);
            }

            pipeline_specifics::store(s, g, inst, i, output_stage);

            if ((i + 1) % iter_scale == 0) {
                for (int j = 0; j < iter_scale; j++) {
                    auto stage_to_arrive =
                        (i - j) % OUTPUT_PIPELINE_STAGES;
                    kittens::warp::arrive(
                        outputs_finished(s, stage_to_arrive));
                }
            }
            output_stage = (output_stage + 1) % OUTPUT_PIPELINE_STAGES;
        }
    }
};

// rms_matvec_pipeline always reduces over full hidden_dim
// (used by QKV, upgate, lm_head).
template <typename Config, typename Globals, typename parsed_instruction,
          typename pipeline_specifics, auto ActPtr, auto RmsPtr>
struct rms_matvec_pipeline
    : public matvec_pipeline<Config, Globals, parsed_instruction,
                             pipeline_specifics> {
    using pipeline = matvec_pipeline<Config, Globals, parsed_instruction,
                                     pipeline_specifics>;

    static constexpr int REDUCTION_DIM_PER_WARP =
        Globals::hidden_dim / Config::NUM_CONSUMER_WARPS;

    static constexpr int SEM_COUNT = 1 + pipeline::SEM_COUNT;

    __device__ static inline kittens::semaphore &
    rms_scale_arrived(megakernel::state<Config> &s) {
        return s.semaphores()[pipeline::SEM_COUNT];
    }

    __device__ static inline kittens::sv_bf<Globals::hidden_dim> &
    get_rms_scale(megakernel::state<Config> &s) {
        return *reinterpret_cast<kittens::sv_bf<Globals::hidden_dim> *>(
            s.pages[pipeline::get_activation_page(s)].ptr(
                sizeof(kittens::sv_bf<Globals::hidden_dim>)));
    }

    __device__ static inline int
    init_semaphores(megakernel::state<Config> &s) {
        pipeline::init_semaphores(s);
        init_semaphore(rms_scale_arrived(s), 1);
        return SEM_COUNT;
    }

    __device__ static inline void
    loader_loop(megakernel::state<Config> &s, const Globals &g,
                int layer_idx) {
        if (kittens::laneid() == 0) {
            int activation_page = pipeline::get_activation_page(s);
            s.wait_page_ready(activation_page);

            auto &rms_scale = get_rms_scale(s);
            auto &sem = rms_scale_arrived(s);

            kittens::tma::expect(sem, rms_scale);
            kittens::tma::load_async<kittens::cache_policy::EVICT_LAST>(
                rms_scale, g.*RmsPtr, {layer_idx, 0}, sem);
        }

        pipeline::loader_loop(s, g);
    }

    __device__ static inline void
    launcher_loop(megakernel::state<Config> &s, const Globals &g) {
        if (kittens::laneid() == 0) {
#ifdef KITTENS_BLACKWELL
            s.wait_tensor_ready();
            arrive(s.tensor_finished, Config::NUM_CONSUMER_WARPS);
#endif
        }
    }

    __device__ static inline void
    consumer_loop(megakernel::state<Config> &s, const Globals &g) {

        using sv_t = kittens::sv_bf<REDUCTION_DIM_PER_WARP>;
        auto &rms_scale_smem =
            reinterpret_cast<sv_t *>(&get_rms_scale(s))[kittens::warpid()];
        auto &activations_smem =
            reinterpret_cast<sv_t *>(
                &pipeline::get_activations(s))[kittens::warpid()];

        if (kittens::laneid() == 0 && kittens::warpid() == 0) {
            parsed_instruction inst{s};

            int activation_page = pipeline::get_activation_page(s);

            s.wait_page_ready(activation_page);
            auto &activations = pipeline::get_activations(s);

            auto &sem = pipeline::activations_arrived(s);

            s.record(megakernel::TEVENT_AT_GMEM_WAIT);
            pipeline_specifics::gmem_wait(g, s);
            s.record(megakernel::TEVENT_DONE_GMEM_WAIT);
        }
        kittens::group<Config::NUM_CONSUMER_WARPS>::sync(3);

        kittens::warp::load(activations_smem, g.*ActPtr,
                            {kittens::warpid()});

        auto activation_page = pipeline::get_activation_page(s);

        kittens::wait(rms_scale_arrived(s), 0);

        // Use the last output stage for RMS scratch (safe: no outputs
        // produced yet).  Using OUTPUT_PIPELINE_STAGES would place it at
        // scratch+3072, which overlaps with the rope cos/sin buffer when
        // head_dim=128 (3B).
        auto activations_vec = rms_norm<Config, Globals>(
            rms_scale_smem, activations_smem, g.rms_norm_eps,
            pipeline::get_output_start(s,
                                       pipeline::OUTPUT_PIPELINE_STAGES - 1));

        kittens::warp::sync();
        s.warp_finish_page(activation_page, 1);

        pipeline::consumer_loop(s, g, activations_vec);
    }
};
