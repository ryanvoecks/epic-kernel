# Mixtral CUDA Megakernel — Implementation Plan

This document is the concrete implementation guide for `demos/mixtral/`. It is
written to be followed file-by-file, referencing the exact patterns from
`demos/low-latency-llama/` wherever reuse is intended.

---

## Directory Layout

```
demos/mixtral/
├── Makefile
├── mixtral.cu                  # pybind11 entry point (like llama_1b.cu)
├── mixtral.cuh                 # globals struct + opcode defines (like llama_1b.cuh)
├── rms_matvec_rope_append.cu   # opcode 1 — copy from llama, adjust dims
├── attention_partial.cu        # opcode 2 — copy from llama, adjust dims
├── attention_reduction.cu      # opcode 3 — copy verbatim (parameterised by globals)
├── matvec_adds.cu              # opcode 4 — copy verbatim (o_proj instantiation)
├── router.cu                   # opcode 5 — NEW
├── expert_upgate.cu            # opcode 6 — NEW
├── expert_downproj.cu          # opcode 7 — NEW
└── rms_lm_head.cu              # opcode 8 — copy from llama, adjust barrier opcode
```

---

## Step 1 — `mixtral.cuh` (globals struct + opcode defines)

This is the first file to write and must compile cleanly before any op files.
Model it directly on `demos/low-latency-llama/llama_1b.cuh`.

### 1.1 Opcode defines

```cpp
#define OPCODE_QKV                  1
#define OPCODE_PartialAttention     2
#define OPCODE_AttentionReduction   3
#define OPCODE_OProj                4
#define OPCODE_MoE_Router           5
#define OPCODE_ExpertUpGateSiLU     6
#define OPCODE_ExpertDownProjAccum  7
#define OPCODE_RMS_LM_Head          8
```

### 1.2 Architecture constants

```cpp
#define MIXTRAL_NUM_LAYERS         32
#define MIXTRAL_HIDDEN_DIM         4096
#define MIXTRAL_INTERMEDIATE_DIM   14336
#define MIXTRAL_HEAD_DIM           128
#define MIXTRAL_NUM_ATTN_HEADS     32
#define MIXTRAL_NUM_KV_HEADS       8
#define MIXTRAL_NUM_EXPERTS        8
#define MIXTRAL_NUM_EXPERTS_TOK    2
#define MIXTRAL_VOCAB_SIZE         32000
#define MIXTRAL_KV_BLOCK_SIZE      16
#define MIXTRAL_MATVEC_BLOCK_SIZE  16
// matvec_reduction_size = gcd(14336, 4096) = 512
#define MIXTRAL_MATVEC_REDUCTION   512
```

### 1.3 Globals struct template

Follow the exact same `struct globals_t<...>` pattern as `llama_1b.cuh`. The key
additions over Llama are the MoE weight and activation layouts.

**Shared layout types (same as Llama, just parameterised on new dims):**

```cpp
// [1, num_layers, row, hidden_dim]  — for qkv, o_proj, router, norm weights
using weights_t         = kittens::gl<kittens::bf16, 1, -1, -1, hidden_dim,
                                       kittens::st_bf<matvec_block_size, 512>>;

// [1, num_layers, row, intermediate_dim]  — for down_proj (wide input dim)
using weights_big_t     = kittens::gl<kittens::bf16, 1, -1, -1, intermediate_dim,
                                       kittens::st_bf<matvec_block_size, 512>>;
```

**New: per-expert weight layouts — flatten layers×experts into depth dim:**

The Python stacked tensors have shape `[num_layers, num_experts, rows, cols]`.
In the CUDA tensor descriptor, flatten `layer * num_experts + expert_idx` into
the depth dimension so that a standard `gl<bf16, 1, -1, -1, col_dim>` descriptor
works. The TMA coordinate is then `{0, layer*num_experts+expert, row, col}`.

```cpp
// expert_gate / expert_up:  [1, num_layers*num_experts, intermediate_dim, hidden_dim]
using expert_proj_t     = kittens::gl<kittens::bf16, 1, -1, -1, hidden_dim,
                                       kittens::st_bf<matvec_block_size, 512>>;

// expert_down:              [1, num_layers*num_experts, hidden_dim, intermediate_dim]
using expert_down_t     = kittens::gl<kittens::bf16, 1, -1, -1, intermediate_dim,
                                       kittens::st_bf<matvec_block_size, 512>>;

// router_weights:           [1, num_layers*num_experts, 1, hidden_dim]
// (only 8 rows per layer — treat as same gl type as weights_t, row count is tiny)
using router_weights_t  = weights_t;
```

**New: activation layouts:**

```cpp
// expert_silu_out:  [1, num_experts, 1, intermediate_dim]
using expert_silu_t = kittens::gl<kittens::bf16, 1, num_experts, 1, intermediate_dim,
                                   kittens::sv_bf<intermediate_dim>>;

// router_normed_hidden: same shape as hidden_states — [1,1,1,hidden_dim]
// Reuse activations_t type.
```

**Runtime router outputs (written by Router op, read by UpGate/DownProj):**

```cpp
// These are tiny — just 2 int32s and 2 floats. Store as plain device pointers
// rather than gl<> layout objects; the controller warp reads them by value.
int32_t *selected_expert_indices;   // device pointer, size num_experts_per_tok
float   *selected_expert_scores;    // device pointer, size num_experts_per_tok
```

**Struct fields (in member order for pybind11):**

```cpp
// megakernel internals
barriers         Bar;
instruction_layout instructions;
timing_layout    timings;

// attention weights (same as Llama)
weights_t        qkv_weights;
norm_weights_t   attn_norm_weights;
weights_t        o_weights;

// MoE weights (new)
expert_proj_t    expert_gate_weights;   // [layers*experts, intermediate, hidden]
expert_proj_t    expert_up_weights;     // [layers*experts, intermediate, hidden]
expert_down_t    expert_down_weights;   // [layers*experts, hidden, intermediate]
router_weights_t router_weights;        // [layers*experts, 1, hidden]  (8 rows/layer)
norm_weights_t   ffn_ln_weights;        // [num_layers, hidden]

// LM head weights
norm_weights_t   lm_head_norm_weights;
weights_t        lm_head_weights;

// KV cache
kv_cache_t       k_cache;
kv_cache_t       v_cache;

// RoPE tables
rope_table_t     rope_cos;
rope_table_t     rope_sin;

// Activation buffers
activations_t    hidden_states;
activations_t    q_post_rope;
activations_t    attn_out;
attn_lse_intermediates_t attn_lse_intermediates;
attn_out_intermediates_t attn_out_intermediates;
expert_silu_t    expert_silu_out;       // [num_experts, intermediate_dim]
activations_t    router_normed_hidden;  // scratch: RMS-normed hidden for UpGate
logits_t         logits;

// Runtime router outputs (plain pointers — not TMA descriptors)
int32_t         *selected_expert_indices;
float           *selected_expert_scores;

// Scalar constants
unsigned int     pos_id;
float            attn_scale;
float            rms_norm_eps;
bool             skip_attn_reduction;
```

**`downproj_reduction_chunk_size` for Mixtral:**

In Llama, `downproj_reduction_chunk_size = hidden_dim` (full reduction in one pass).
For Mixtral, col-splits are `MIXTRAL_MATVEC_REDUCTION = 512` wide, so:

```cpp
constexpr static int downproj_reduction_chunk_size = matvec_reduction_size; // 512
```

This changes the `EXPECTED_ARRIVAL_COUNT` for QKV (waiting on ExpertDownProj of
previous layer). The QKV op waits until:

```
num_experts_per_tok * (intermediate_dim / matvec_reduction_size) * (hidden_dim / matvec_block_size)
= 2 * 28 * 256 = 14336
```

In the globals struct, add a constexpr for the QKV arrival count:

```cpp
constexpr static int qkv_expected_arrivals =
    num_experts_per_tok *
    (intermediate_dim / downproj_reduction_chunk_size) *
    (hidden_dim / matvec_block_size);
// = 2 * 28 * 256 = 14336
```

**Typedef:**

```cpp
typedef globals_t<MIXTRAL_NUM_LAYERS, MIXTRAL_HIDDEN_DIM, MIXTRAL_INTERMEDIATE_DIM,
                  MIXTRAL_HEAD_DIM, MIXTRAL_NUM_ATTN_HEADS, MIXTRAL_NUM_KV_HEADS,
                  MIXTRAL_NUM_EXPERTS, MIXTRAL_NUM_EXPERTS_TOK,
                  MIXTRAL_KV_BLOCK_SIZE, MIXTRAL_MATVEC_BLOCK_SIZE,
                  MIXTRAL_MATVEC_REDUCTION, B200_SM_COUNT>
    mixtral_globals;
```

---

## Step 2 — Reused ops (opcodes 1–4, 8)

Each file needs only two changes versus `demos/low-latency-llama/`:
1. Replace `#include "llama.cuh"` with `#include "mixtral.cuh"`
2. Replace `using globals = llama_globals;` with `using globals = mixtral_globals;`

Then fix any hardcoded dim references (e.g. `llama_globals::hidden_dim`) to use
the template parameter.

### `rms_matvec_rope_append.cu` (opcode 1)

- `EXPECTED_ARRIVAL_COUNT` = `qkv_expected_arrivals` from globals struct
  (previously was `(intermediate_dim / downproj_reduction_chunk_size) * (hidden_dim / matvec_block_size)` × 1 expert; now multiply by `num_experts_per_tok`)
- Everything else (K/V block boundary detection, RoPE logic) is identical —
  `head_dim=128` vs 64 is handled by the template parameter

### `attention_partial.cu` (opcode 2)

- Copy verbatim. The only relevant constants are `num_kv_heads=8`, `head_dim=128`,
  `kv_block_size=16` — all are template parameters already.

### `attention_reduction.cu` (opcode 3)

- Copy verbatim. Fully parameterised on `Globals::num_attention_heads`,
  `Globals::num_kv_heads`, `Globals::head_dim`, `Globals::sm_count`.

### `matvec_adds.cu` (opcode 4 — o_proj)

- Copy verbatim. The `o_proj` instantiation at the bottom reads:
  ```cpp
  template <typename Config, typename Globals>
  struct o_proj : MatVecAddOp<
      Globals::num_attention_heads,          // EXPECTED (= 32 for Mixtral)
      &Globals::o_weights, &Globals::attn_out,
      &Globals::hidden_states, OPCODE_OProj,
      OPCODE_OProj - 1, Config, Globals
  > {};
  ```
  No changes needed beyond the include swap.

### `rms_lm_head.cu` (opcode 8)

- `EXPECTED_ARRIVAL_COUNT` = `Globals::qkv_expected_arrivals` (same formula as QKV)
- Change the barrier wait to reference `OPCODE_ExpertDownProjAccum` (opcode 7)
  instead of `OPCODE_DownProjResidual`:
  ```cpp
  while (*(volatile int *)&g.Bar[{Globals::num_layers - 1,
                                   OPCODE_ExpertDownProjAccum - 1, 0}] <
         EXPECTED_ARRIVAL_COUNT) { ... }
  ```

---

## Step 3 — `router.cu` (opcode 5) — NEW

### What it does

1. Waits for all o_proj blocks to finish (`Bar[layer, OPCODE_OProj-1, 0] >= num_attn_heads`)
2. RMS norm of `hidden_states` → writes `router_normed_hidden`
3. Router matVec: `router_weights[layer * num_experts .. +num_experts] @ normed_hidden`
   → 8 scalars (one per expert). This is tiny (8×4096) — use the scalar register path
4. Softmax over 8 scalars + top-2 selection + renormalize
5. Write `selected_expert_indices[0..1]` and `selected_expert_scores[0..1]`
6. Signal completion: `Bar[layer, OPCODE_MoE_Router-1, 0] = 1`

### Instruction layout (serialised words)

```
word[0] = opcode (5)
word[1] = layer_idx
words[2..31] = unused (zero)
```

### Warp roles

The router is single-SM and the matVec is tiny. The simplest implementation
uses the **consumer warp group** exclusively:

- **controller**: parse instruction, check o_proj barrier, signal start
- **loader**: TMA load `router_weights[layer * num_experts..(layer+1)*num_experts, :, :]`
  and `ffn_ln_weights[layer]` into shared memory (small — 8×4096×2 bytes = 65KB;
  fits only if we do it row-by-row or use a non-TMA path for this tiny op)
- **consumer**: RMS norm, 8-row matVec loop, softmax, top-2, write outputs
- **storer**: write `router_normed_hidden` to global via TMA, set barrier

### RMS norm implementation

Reuse the same RMS norm pattern as `rms_matvec_rope_append.cu`. The normed
vector must be stored in `router_normed_hidden` (global memory) so that all
subsequent `ExpertUpGateSiLU` SMs can read it.

### Softmax + top-2

All 8 logits fit in registers. Use `__expf`, find max, accumulate, divide.
`top-2` is a simple double-max scan. All computation happens in one warp;
the result is written to device memory via `st.global`.

### Barrier output

```cpp
// Using atomicExch (not atomicAdd) since only one SM writes this:
atomicExch(&g.Bar[{layer_idx, OPCODE_MoE_Router - 1, 0}], 1);
```

---

## Step 4 — `expert_upgate.cu` (opcode 6) — NEW

### What it does

For a given `(layer_idx, expert_idx)`:
1. Check if `expert_idx ∈ selected_expert_indices` — if not, **skip** (fast path)
2. Wait for `Bar[layer, OPCODE_MoE_Router-1, 0] == 1`
3. Load blocks of `expert_gate_weights[layer*num_experts+expert, block, :]` and
   `expert_up_weights[layer*num_experts+expert, block, :]` via TMA
4. MatVec each row-block against `router_normed_hidden`
5. SiLU(gate) * up → write to `expert_silu_out[expert, block*BS..(block+1)*BS]`
6. Signal: `Bar[layer, OPCODE_ExpertUpGateSiLU-1, expert_idx] += num_blocks`

### Instruction layout

```
word[0] = opcode (6)
word[1] = layer_idx
word[2] = expert_idx
word[3] = start_block_idx
word[4] = num_blocks       (= end_block_idx - start_block_idx)
words[5..31] = unused
```

### Design: mirrors `upgate.cu` (`rms_upgate_silu`)

The Llama `rms_upgate_silu` op (in `upgate.cu`) already does the double-matVec
(gate + up) with SiLU using `matvec_pipeline`. The Mixtral `expert_upgate` is
the same op with two changes:

1. **Expert dimension in weight coords**: replace `{inst.layer_idx, block_idx, col_idx}`
   with `{0, inst.layer_idx * num_experts + inst.expert_idx, block_idx * matvec_block_size, col_idx * 512}`.
   Using `expert_proj_t` (depth = layers×experts).

2. **Conditional skip**: in `controller::release_lid` or `pipeline::gmem_wait`,
   check `selected_expert_indices` before issuing loads:
   ```cpp
   bool active = false;
   for (int i = 0; i < Globals::num_experts_per_tok; i++) {
       if (g.selected_expert_indices[i] == inst.expert_idx) { active = true; break; }
   }
   if (!active) {
       // increment barrier so DownProj can proceed, then return
       atomicAdd(&g.Bar[{inst.layer_idx, OPCODE_ExpertUpGateSiLU - 1, inst.expert_idx}],
                 inst.num_blocks);
       return;
   }
   ```

3. **Input vector**: `router_normed_hidden` instead of running RMS norm in-kernel
   (the router already wrote the normed vector to global memory).

4. **Output**: `expert_silu_out[expert_idx, block]` instead of `silu_out[block]`.
   TMA store coord: `{0, inst.expert_idx, 0, inst.start_block_idx * matvec_block_size}`
   using the `expert_silu_t` layout.

### `parsed_instruction`

```cpp
struct parsed_instruction {
    int layer_idx, expert_idx, start_block_idx, num_blocks, iters;
    int depth;   // = layer_idx * num_experts + expert_idx (precomputed)
    __device__ inline parsed_instruction(typename Config::instruction_t &instr) {
        layer_idx      = instr[1];
        expert_idx     = instr[2];
        start_block_idx = instr[3];
        num_blocks     = instr[4];
        iters          = 2 * num_blocks;  // gate + up interleaved (like Llama upgate)
        depth          = layer_idx * Globals::num_experts + expert_idx;
    }
};
```

### `EXPECTED_ARRIVAL_COUNT` (for router barrier check)

```cpp
static constexpr int EXPECTED_ARRIVAL_COUNT = 1; // router sets Bar[layer,4,0]=1
```

---

## Step 5 — `expert_downproj.cu` (opcode 7) — NEW

### What it does

For a given `(layer_idx, expert_idx, start_block_idx..end_block_idx, reduction_block_idx)`:
1. Check if `expert_idx ∈ selected_expert_indices` — if not, **skip**
2. Wait for `Bar[layer, OPCODE_ExpertUpGateSiLU-1, expert_idx] == expected_upgate_count`
3. Load blocks of `expert_down_weights[layer*num_experts+expert, block, :]` via TMA
4. MatVec against `expert_silu_out[expert_idx, reduction_block_idx*512..]`
5. Look up `weight = selected_expert_scores[slot]` for this expert
6. Weighted-add: `hidden_states[block] += weight * result`
7. Signal: `Bar[layer, OPCODE_ExpertDownProjAccum-1, 0] += num_blocks`

### Instruction layout

```
word[0] = opcode (7)
word[1] = layer_idx
word[2] = expert_idx
word[3] = start_block_idx
word[4] = end_block_idx
word[5] = reduction_block_idx
words[6..31] = unused
```

### Design: based on `MatVecAddOp` / `downproj` from `matvec_adds.cu`

The Llama `downproj` in `matvec_adds.cu` is instantiated as `MatVecAddOp` with:
- `WeightsPtr = &Globals::down_weights`
- `InputActivationsPtr = &Globals::silu_out`
- `OutputActivationsPtr = &Globals::hidden_states`
- `REDUCTION_DIM = downproj_reduction_chunk_size`
- Uses `tma::store_add_async` (atomic add) for the output accumulation

The Mixtral version reuses this same `MatVecAddOp` template but with:

1. **Expert weight indexing** via `expert_down_t`: depth coord = `layer * num_experts + expert`
2. **Per-expert `silu_out` input**: coord `{0, expert_idx, 0, reduction_start_col}`
3. **Weighted output**: the `store` step must multiply by `selected_expert_scores[slot]`
   before calling `tma::store_add_async`

Because the weighting requires a scalar multiply before the store, and `MatVecAddOp`'s
`store` function doesn't accept a scale factor, the cleanest approach is a thin wrapper
struct that overrides `pipeline_specifics::store`:

```cpp
template <typename Config, typename Globals>
struct expert_downproj {
    // ... (same structure as MatVecAddOp but with customised store)
    struct pipeline_specifics {
        // load_iter: identical to MatVecAddOp but with expert depth coord
        template <int TW>
        static __device__ inline void
        load_iter(state<Config> &s, const Globals &g, parsed_instruction &inst,
                  int iter, int col_idx, kittens::st_bf<16,TW> &chunk, kittens::semaphore &sem) {
            kittens::tma::load_async<dim::ROW, cache_policy::EVICT_FIRST>(
                chunk, g.expert_down_weights,
                coord<>{0, inst.depth,
                        (inst.start_block_idx + iter) * Globals::matvec_block_size,
                        inst.start_reduction_col + TW * col_idx},
                sem);
        }

        static __device__ inline void
        store(state<Config> &s, const Globals &g, parsed_instruction &inst,
              int output_idx, int output_stage) {
            int block_idx = inst.start_block_idx + output_idx;

            // Reduce partial matVec result
            kittens::rv_fl<16> output_rv;
            matvec_reduce<...>(get_output_start(s, output_stage), output_rv);

            // Look up expert weight scalar
            float expert_weight = 0.f;
            for (int i = 0; i < Globals::num_experts_per_tok; i++) {
                if (g.selected_expert_indices[i] == inst.expert_idx) {
                    expert_weight = g.selected_expert_scores[i];
                    break;
                }
            }

            // Scale output by expert weight
            kittens::warp::mul(output_rv, output_rv, expert_weight);

            kittens::sv_bf<16> output_smem;
            kittens::warp::store(output_smem, output_rv);
            kittens::warp::sync();

            if (kittens::warp::laneid() == 0) {
                // Atomic add into hidden_states (safe: serialised across experts by DAG)
                kittens::tma::store_add_async<cache_policy::EVICT_LAST>(
                    g.hidden_states, output_smem, {block_idx});
                kittens::tma::store_async_read_wait();
            }
        }
    };
    // ... controller, loader, consumer, storer follow MatVecAddOp pattern
};
```

### `EXPECTED_ARRIVAL_COUNT` (for UpGate barrier check)

```cpp
// UpGate increments Bar[layer, 5, expert_idx] by num_blocks per instruction.
// Total arrivals for one expert = intermediate_dim / matvec_block_size = 14336/16 = 896.
static constexpr int EXPECTED_ARRIVAL_COUNT =
    Globals::intermediate_dim / Globals::matvec_block_size;  // 896
```

### Barrier output

```cpp
atomicAdd(&g.Bar[{inst.layer_idx, OPCODE_ExpertDownProjAccum - 1, 0}],
          inst.end_block_idx - inst.start_block_idx);
```

---

## Step 6 — `mixtral.cu` (pybind11 entry point)

```cpp
#include "pyutils/pyutils.cuh"
#include "mixtral.cuh"

#include "rms_matvec_rope_append.cu"   // opcode 1
#include "attention_partial.cu"        // opcode 2
#include "attention_reduction.cu"      // opcode 3
#include "matvec_adds.cu"              // opcode 4
#include "router.cu"                   // opcode 5
#include "expert_upgate.cu"            // opcode 6
#include "expert_downproj.cu"          // opcode 7
#include "rms_lm_head.cu"              // opcode 8

using namespace kittens;
using namespace megakernel;

using mixtral_qkv_op       = rms_qkv_rope_append<default_config, mixtral_globals>;
using mixtral_pattn_op     = attention_partial<default_config, mixtral_globals>;
using mixtral_attn_red_op  = attention_reduction<default_config, mixtral_globals>;
using mixtral_oproj_op     = o_proj<default_config, mixtral_globals>;
using mixtral_router_op    = moe_router<default_config, mixtral_globals>;
using mixtral_upgate_op    = expert_upgate<default_config, mixtral_globals>;
using mixtral_downproj_op  = expert_downproj<default_config, mixtral_globals>;
using mixtral_lmhead_op    = rms_lm_head<default_config, mixtral_globals>;

PYBIND11_MODULE(mk_mixtral, m) {
    m.doc() = "";
    kittens::py::bind_kernel<
        mk<default_config, mixtral_globals,
           mixtral_pattn_op, mixtral_attn_red_op, mixtral_qkv_op,
           mixtral_downproj_op, mixtral_oproj_op,
           mixtral_router_op, mixtral_upgate_op,
           mixtral_lmhead_op>>(m, "mk_mixtral",
        // megakernel internals
        &mixtral_globals::Bar,
        &mixtral_globals::instructions,
        &mixtral_globals::timings,
        // attention weights
        &mixtral_globals::qkv_weights,
        &mixtral_globals::attn_norm_weights,
        &mixtral_globals::o_weights,
        // MoE weights
        &mixtral_globals::expert_gate_weights,
        &mixtral_globals::expert_up_weights,
        &mixtral_globals::expert_down_weights,
        &mixtral_globals::router_weights,
        &mixtral_globals::ffn_ln_weights,
        // LM head
        &mixtral_globals::lm_head_norm_weights,
        &mixtral_globals::lm_head_weights,
        // KV cache + RoPE
        &mixtral_globals::k_cache,
        &mixtral_globals::v_cache,
        &mixtral_globals::rope_cos,
        &mixtral_globals::rope_sin,
        // activations
        &mixtral_globals::hidden_states,
        &mixtral_globals::q_post_rope,
        &mixtral_globals::attn_out,
        &mixtral_globals::attn_lse_intermediates,
        &mixtral_globals::attn_out_intermediates,
        &mixtral_globals::expert_silu_out,
        &mixtral_globals::router_normed_hidden,
        &mixtral_globals::logits,
        // router runtime outputs
        &mixtral_globals::selected_expert_indices,
        &mixtral_globals::selected_expert_scores,
        // scalars
        &mixtral_globals::pos_id,
        &mixtral_globals::attn_scale,
        &mixtral_globals::rms_norm_eps,
        &mixtral_globals::skip_attn_reduction);
}
```

The argument order here **must exactly match** the order in `MixtralGlobals` in
`megakernels/mixtral/instructions.py` and the `make_globals` function in
`megakernels/mixtral/scheduler.py`, since pybind11 maps tensors positionally.

---

## Step 7 — `Makefile`

Copy `demos/low-latency-llama/Makefile` and change:
- `llama.cu` → `mixtral.cu`
- Output target: `mk_mixtral$(EXT_SUFFIX)`
- Flags: same nvcc flags

```makefile
mk_mixtral: mixtral.cu
    mkdir -p $(BUILD_DIR)
    $(NVCC) mixtral.cu $(NVCC_FLAGS) -o $(BUILD_DIR)/mk_mixtral$(EXT_SUFFIX)
```

---

## Step 8 — Update `megakernels/mixtral/mk.py`

Once the kernel is built, replace the `NotImplementedError` stub:

```python
class MixtralMK_Interpreter(MK_Interpreter):
    def __init__(self, mk_dir: Path):
        import importlib.util, sys
        so_files = list(mk_dir.glob("mk_mixtral*.so"))
        if not so_files:
            raise FileNotFoundError(f"mk_mixtral .so not found in {mk_dir}")
        spec = importlib.util.spec_from_file_location("mk_mixtral", so_files[0])
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)

    def interpret(self, globs: MixtralGlobals):
        self.mod.mk_mixtral(
            globs.barriers,
            globs.instructions,
            globs.timings,
            # ... (same order as pybind11 binding)
        )
```

Look at `megakernels/demos/latency/mk.py` for the exact `interpret()` call
pattern and replicate it for Mixtral's fields.

---

## Barrier Layout Reference

The Python scheduler allocates:
```
barriers: [num_layers, 10, max(num_attn_heads + 2*num_kv_heads, num_experts)]
         = [32, 10, max(48, 8)] = [32, 10, 48]
```

Barrier slot assignments (0-indexed opcode → slot `opcode-1`):

| Slot | Opcode | Filled by | Waited by | Value when done |
|------|--------|-----------|-----------|-----------------|
| 0    | QKV (1) | `solve_qkv`: +1 per head block | PartialAttn: checks per-head | `bph` per head |
| 1    | PartialAttn (2) | `solve_partial_attn`: +1 per kv_head | AttnReduction / OProj (skip_attn_reduction) | `num_partitions` |
| 2    | AttnReduction (3) | `solve_attn_reduction`: +gqa_ratio per kv_head | OProj | `num_attn_heads` |
| 3    | OProj (4) | `solve_oproj`: +1 per block | Router: checks slot[0] | `num_o_blocks` |
| 4    | Router (5) | `solve_router`: set slot[0]=1 | ExpertUpGate: checks slot[0] | 1 |
| 5    | ExpertUpGate (6) | `solve_expert_upgate`: +num_blocks at `slot[expert_idx]` | ExpertDownProj: checks `slot[expert_idx]` | `intermediate_dim / block_size` per expert |
| 6    | ExpertDownProj (7) | `solve_expert_downproj`: +blocks at `slot[0]` | Next-layer QKV + LM Head | `num_experts_per_tok * col_splits * down_blocks` |

---

## Implementation Order

Work through files in this order, testing each before moving on:

1. **`mixtral.cuh`** — globals struct. Verify it compiles with an empty `mixtral.cu`.
2. **Reused ops (1-4, 8)** — copy and fix includes. Compile; run `diff_test_mixtral.py stop_after_op=oproj layer_limit=1` with PyVM-only.
3. **`router.cu`** — implement and compile. Run `diff_test_mixtral.py stop_after_op=router layer_limit=1`.
4. **`expert_upgate.cu`** — implement and compile. Run `diff_test_mixtral.py stop_after_op=expert_upgate layer_limit=1 force_experts=0,1`.
5. **`expert_downproj.cu`** — implement and compile. Run full single-layer diff test.
6. **Update `mk.py`** — wire up the interpreter. Run `mode=mk setting=mixtral_latency`.

---

## Key Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| `expert_silu_out` TMA layout — `[num_experts, intermediate_dim]` is 8×14336×2=229KB, exceeds one page | Split loads by expert slice; each UpGate instruction only writes `num_blocks * 16` rows |
| Hidden state write race in DownProj | DAG serialises all DownProj for the same (expert, col_split) across SMs; confirmed by Python scheduler |
| `selected_expert_indices` as raw pointer vs TK layout | Pass as `int32_t*` to pybind11 (not a `gl<>`); read in-kernel with plain `ld.global` |
| Large `EXPECTED_ARRIVAL_COUNT = 14336` for QKV | Correct by construction — the spin loop will see the full count once 2 active experts × 28 col-splits × 256 blocks complete |
| `head_dim=128` vs 64 — RoPE scratch may overflow | `Config::SCRATCH_BYTES` holds `2 × head_dim × sizeof(float)` = 1KB for 128-dim; verify against config |
