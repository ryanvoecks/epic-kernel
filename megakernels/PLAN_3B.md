# Plan: Llama 3B MegaKernel Support

## Overview

This document describes all steps needed to extend the MegaKernel framework to support **Llama 3.2 3B** (`meta-llama/Llama-3.2-3B-Instruct`) for low-latency single-token decode. The implementation will live in `megakernels/llama_3b/` and `demos/low-latency-llama-3b/` as new files, leaving the existing Llama 1B code untouched.

---

## 1. Model Dimension Changes (1B → 3B)

| Parameter | Llama 1B | Llama 3B | Impact |
|-----------|----------|----------|--------|
| `num_hidden_layers` | 16 | 28 | 1.75× more layers in the DAG; schedule length grows proportionally |
| `hidden_size` | 2048 | 3072 | 1.5× wider matvecs; 1.5× the activation buffer; RMS norm divides by 3072 |
| `intermediate_size` | 8192 | 8192 | Same value, but `8192 / 3072` is **not an integer** — column-split strategy must change |
| `head_dim` | 64 | 128 | 2× wider Q/K/V heads; changes RoPE tile; changes attention tile sizes |
| `num_attention_heads` | 32 | 24 | Fewer Q heads; GQA ratio changes from 4 to 3 |
| `num_key_value_heads` | 8 | 8 | Same |
| `vocab_size` | 128256 | 128256 | Same |
| `max_position_embeddings` | 131072 | 131072 | Same |
| `rms_norm_eps` | 1e-5 | 1e-5 | Same |

**Key challenges**:
1. `intermediate_size = 8192` is the same as 1B, but `8192 / 3072 ≈ 2.67`, so the column-split strategy for down-projection must change compared to the 1B case where `8192 / 2048 = 4` evenly.
2. `GQA_RATIO = num_attention_heads / num_kv_heads = 24 / 8 = 3` (was 4). This affects attention code that groups Q heads per KV head.

---

## 2. File Structure

### New Python-side files (`megakernels/llama_3b/`)

```
megakernels/llama_3b/
    __init__.py
    instructions.py    # Re-exports from latency module (same opcodes, different sizes at runtime)
    scheduler.py       # 3B-specific ScheduleBuilder, updated DAG construction
    python_vm.py       # 3B reference interpreter (barrier counts derived from globals)
    mk.py              # 3B MK_Interpreter calling mk_llama_3b pybind module
```

### New CUDA files (`demos/low-latency-llama-3b/`)

```
demos/low-latency-llama-3b/
    Makefile
    llama_3b.cu        # Entry point, pybind11 bindings for llama_3b_globals
    llama_3b.cuh       # globals_t typedef for 3B dims, forward declarations
    utils.cuh          # Updated rms_norm (divides by 3072 instead of 2048)
    matvec_pipeline.cuh  # Updated REDUCTION_DIM_PER_WARP for hidden_dim=3072
    matvec_adds.cu     # o_proj and downproj ops, updated arrival counts
    rms_matvec_rope_append.cu  # QKV op, updated K/V block boundaries for head_dim=128
    attention_partial.cu       # Updated tile sizes for head_dim=128, GQA_RATIO=3
    attention_reduction.cu     # Updated tile sizes for head_dim=128, GQA_RATIO=3
    upgate.cu          # Updated expected arrival count for hidden_dim=3072
    rms_lm_head.cu     # Updated expected arrival count for 28 layers
```

### Dispatch integration (edit existing files)

- `megakernels/dispatch.py` — register `"latency_3b"` setting

---

## 3. Detailed Implementation Steps

### Step 1: Create `llama_3b.cuh` — CUDA globals header

Define 3B constants and the `llama_3b_globals` typedef:

```cpp
#define LLAMA_3B_NUM_LAYERS 28
#define LLAMA_3B_HIDDEN_DIM 3072
#define LLAMA_3B_INTERMEDIATE_DIM 8192
#define LLAMA_3B_HEAD_DIM 128
#define LLAMA_3B_NUM_ATTENTION_HEADS 24
#define LLAMA_3B_NUM_KV_HEADS 8
#define LLAMA_3B_KV_BLOCK_SIZE 16
#define LLAMA_3B_MATVEC_BLOCK_SIZE 16
#define LLAMA_3B_LM_HEAD_BLOCK_SIZE 32
#define LLAMA_3B_VOCAB_SIZE 128256

typedef globals_t<LLAMA_3B_NUM_LAYERS, LLAMA_3B_HIDDEN_DIM,
                  LLAMA_3B_INTERMEDIATE_DIM, LLAMA_3B_HEAD_DIM,
                  LLAMA_3B_NUM_ATTENTION_HEADS, LLAMA_3B_NUM_KV_HEADS,
                  LLAMA_3B_KV_BLOCK_SIZE, LLAMA_3B_MATVEC_BLOCK_SIZE,
                  H100_SM_COUNT>  // or B200_SM_COUNT
    llama_3b_globals;
```

**Critical changes from 1B header**:
- The `globals_t` struct uses `hidden_dim` as the inner dimension for `weights_t`, `activations_t`, `norm_weights_t`, etc. — with `hidden_dim=3072`, these all grow by 1.5×.
- `weights_big_indim_t` uses `intermediate_dim=8192` — same as 1B, so TMA layout is unchanged for up/gate/down projections.
- `rope_table_t` uses `head_dim=128` — this increases the rope table from 64 to 128 floats per position.
- `kv_cache_t` uses `head_dim=128` — KV blocks are now 16×128 = 2048 bytes each (2× larger).
- `num_attention_heads=24` — affects QKV out-dim and attention head grouping.

**Shared memory impact**:
- `activations_t` at 3072 bf16 = 6144 bytes per activation vector. A single page (16384 bytes) can hold one activation + one RMS scale vector easily.
- The `matvec_pipeline` loads `st_bf<16, 512>` weight tiles; with `hidden_dim=3072`, `REDUCTION_DIM_PER_WARP = 3072/16 = 192` (was 128).
- The 4-page pipeline stage with `st_bf<16, 512>` covers `4 × 512 = 2048` elements per stage. With `hidden_dim=3072`, we need `ceil(3072/2048) = 2` load rounds per matvec iteration, or `STAGE_PAGES = 6` to cover all 3072 cols in one go (6 × 512 = 3072).

### Step 2: Restructure `matvec_pipeline.cuh` for 3072 hidden dim

The core matvec pipeline currently assumes:
- `REDUCTION_DIM_PER_WARP = hidden_dim / NUM_CONSUMER_WARPS = 2048 / 16 = 128`
- `STAGE_PAGES = 4` — each page holds a `st_bf<16, 512>` tile, so 4 pages = 2048 cols = full hidden_dim

With `hidden_dim=3072`:
- `REDUCTION_DIM_PER_WARP = 3072 / 16 = 192`
- Need `STAGE_PAGES = 6` to cover 3072 columns (6 pages × 512 = 3072)
- With `NUM_PAGES = 13` total, and `INPUT_PIPELINE_STAGES`:
  - 3 stages: `1 (activation) + 3×6 = 19` pages — too many
  - 2 stages: `1 + 2×6 = 13` pages — exactly fits!
  - 1 stage: `1 + 6 = 7` pages — fits with room to spare

**Recommended approach**: Use `INPUT_PIPELINE_STAGES = 2` with `STAGE_PAGES = 6`. This gives exactly 13 pages used. If register pressure is an issue, fall back to 1 stage (7 pages, leaving 6 for output stages).

Note: 3072 is divisible by 512 (`3072 / 512 = 6`), so `st_bf<16, 512>` tiles align perfectly — no TMA alignment issues.

### Step 3: Update `rms_matvec_rope_append.cu`

Key changes:
- **RMS norm divisor**: `variance = full_sum / 3072.0f` (was `/2048.0f` — currently hardcoded in `utils.cuh`)
- **K/V block boundaries**: With QKV out-dim = `(24 + 2×8) × 128 = 5120`:
  - `K_BLK_START = 24 × 128 / 16 = 192` (was `2048/16 = 128`)
  - `V_BLK_START = (24 + 8) × 128 / 16 = 256` (was `2560/16 = 160`)
  - Total QKV blocks = `5120 / 16 = 320` (was `3072/16 = 192`)
- **RoPE**: `head_dim=128` means the interleaved RoPE operation works on 128-element vectors instead of 64. The current code does RoPE via `__shfl_sync` with `laneid() < 16` (processing 16 pairs = 32 elements of a 64-dim head). For 128-dim heads, each warp lane touches more elements; the code must be generalized.
  - With `matvec_block_size=16` and `head_dim=128`, each block covers `16/128 = 1/8` of a head. So `head_chunk = block_idx % 8` (was `% 4`). RoPE cos/sin scratch must hold 128 floats = 512 bytes (was 256).
- **Rope scratch memory**: Currently uses last 512 bytes of scratch for cos (256 bytes for 64 floats) + sin (256 bytes). With head_dim=128, we need 1024 bytes total. Verify scratch has room.
- **Expected arrival count for down_proj predecessor**: Depends on `matvec_reduction_size`. With `reduction_size=1024`: total downproj barrier count = `(8192/1024) × (3072/16) = 8 × 192 = 1536`.

### Step 4: Update `attention_partial.cu`

Key changes:
- **Tile types** change with `head_dim=128`:
  - `q_rt = rt_bf<16, 128>` (was `<16, 64>`) — 2× wider
  - `q_st = st_bf<16, 128>` — 2× wider (4096 bytes → fits in one page)
  - `k_rt = rt_bf<16, 128>` — note KV_BLOCK_SIZE stays at 16
  - `v_rt = rt_bf<16, 128, col_l>`
  - `kv_st = st_bf<16, 128>` — 4096 bytes per KV tile (was 2048)
  - `attn_fl_rt = rt_fl<16, 16>` — unchanged (attention scores: 16 Q-positions × 16 KV-positions)
  - `o_rt = rt_fl<16, 128>` — 2× wider output accumulator
- **Shared memory layout**: Q page needs `st_bf<16,128>` = 4096 bytes + 3 × `sv_fl<128>` = 3×512 = 1536 bytes + `sv_fl<16>` = 64 bytes ≈ 5696 bytes. Fits in one 16KB page.
  - KV page needs `NUM_STAGES × 2 × st_bf<16,128>` = `3 × 2 × 4096 = 24576 bytes` → **exceeds** one 16KB page! Must either reduce `NUM_STAGES` to 2 (gives `2 × 2 × 4096 = 16384`, barely fits) or use two pages.
- **Q loading (`load_Q_async`)**: Currently hardcodes `LLAMA_1B_HEAD_DIM == 64 && GQA_RATIO == 4`. For 3B, `head_dim=128` and `GQA_RATIO=3`. Each Q head is 128 elements = 256 bytes. Loading 3 heads = 768 bytes. A warp has 32 threads; the current scheme loads 32 × 16 = 512 bytes per iteration. For 768 bytes we need 2 rounds (512 + 256).
- **GQA_RATIO**: Now 3 (24 Q / 8 KV). The `store_4_rows` helper must become `store_3_rows`. Per-warp head assignment changes: each KV head maps to 3 Q heads instead of 4.

### Step 5: Update `attention_reduction.cu`

Key changes:
- `o_sv = sv_fl<128>` (was `<64>`) — 512 bytes per head (was 256)
- `o_final_sv = sv_bf<128>` (was `<64>`) — 256 bytes
- `SMEM_PER_HEAD` and `SMEM_PER_STAGE` grow → `NUM_STAGES` decreases
- `Q_HEADS_PER_INSTRUCTION` changes to 3 (matched to GQA_RATIO)
- Consumer warps accumulate 128-wide vectors instead of 64

### Step 6: Update `matvec_adds.cu` (o_proj and downproj)

**O-projection**:
- Input: `attn_out` (3072 bf16) → output: `hidden_states` (3072 bf16)
- Weight: `o_proj` is (3072, 3072) — same in/out dim for 3B
- `EXPECTED_ARRIVAL_COUNT` for o_proj: `num_attention_heads = 24` (was 32)
- The `MatVecAddOp` template references `Globals::hidden_dim` and `Globals::matvec_block_size`, which auto-adjust

**Down-projection**:
- Weight: `down_proj` is (3072, 8192) — `hidden_dim × intermediate_dim`
- `EXPECTED_ARRIVAL_COUNT` for downproj: `hidden_dim / matvec_block_size = 3072 / 16 = 192` (was `2048/16 = 128`)
- The `MatVecAddOp` uses `Globals::hidden_dim` for the arrival count, so the template should auto-adjust
- **Key issue**: `8192 / 3072 ≈ 2.67` is not an integer. The scheduler uses `num_col_splits = intermediate_size // matvec_reduction_size`.
  - **Solution**: Set `matvec_reduction_size = 1024`. Then `num_col_splits = 8192/1024 = 8`. Each SM processes a 16-row × 1024-col slice. The CUDA kernel uses `reduction_block_idx * reduction_size` as start column. `1024` also divides `hidden_size = 3072` evenly (`3072/1024 = 3`), so o_proj column splits also work.

### Step 7: Update `upgate.cu`

- `EXPECTED_ARRIVAL_COUNT = hidden_dim / matvec_block_size = 3072 / 16 = 192` (was 128)
- The `rms_upgate_silu` op stores via TMA to `silu_out`, which is 8192 bf16 wide (same as 1B)
- Barrier update: `block_idx * matvec_block_size / hidden_dim` — with 3B this gives `block_idx * 16 / 3072`. Total up/gate blocks = `8192 / 16 = 512` (same as 1B).

### Step 8: Update `rms_lm_head.cu`

- `EXPECTED_ARRIVAL_COUNT` for down_proj: depends on reduction strategy. With `matvec_reduction_size=1024`: total = `(8192/1024) × (3072/16) = 8 × 192 = 1536`
- The `rms_lm_head_op` waits on layer `num_layers - 1 = 27` (was 15)
- LM head weight is (128256, 3072) — with `lm_head_block_size=32`, that's `128256/32 = 4008` blocks
- The matvec pipeline processes the 3072-wide hidden dim, same structural logic as QKV

### Step 9: Update `utils.cuh` — RMS norm

The `rms_norm` function hardcodes the divisor:
```cpp
float variance = full_sum / 2048.0f;
```
Must change to:
```cpp
float variance = full_sum / static_cast<float>(Globals::hidden_dim);
```
Make `rms_norm` a template that takes the hidden dim as a parameter, or pass it as a runtime argument.

### Step 10: Create `llama_3b.cu` — pybind11 entry point

Similar to `llama.cu` but using `llama_3b_globals`:
```cpp
using rms_qkv_rope_append_op = rms_qkv_rope_append<default_config, llama_3b_globals>;
// ... etc

PYBIND11_MODULE(mk_llama_3b, m) {
    kittens::py::bind_kernel<
        mk<default_config, llama_3b_globals, ...>>(m, "mk_llama_3b", ...);
}
```

All `&llama_1b_globals::` references become `&llama_3b_globals::`.

### Step 11: Create `Makefile` for 3B

Copy the 1B Makefile, change:
```makefile
TARGET ?= mk_llama_3b
SRC ?= llama_3b.cu
```

### Step 12: Create `megakernels/llama_3b/instructions.py`  ✅ DONE

Re-exports from the latency module. The `Globals` dataclass is shared — only runtime values differ:

```python
# Re-export all instruction types from the latency module.
from megakernels.demos.latency.instructions import (
    Globals, AttentionReduction, DownProjResidual,
    LayerNorm_QKV_MatVecRopeAppend, LayerNormDoubleMatVecSiLU,
    O_ProjResidual, PartialAttention, RMS_LM_Head,
)
```

At runtime, `Globals` fields are populated with 3B shapes:
- `post_ln_rope_q: Tensor       # shape: (3072,)`
- `attn_out: Tensor              # shape: (3072,)`
- `attn_lse_intermediates: Tensor  # shape: (24, max_partitions)`
- `attn_out_intermediates: Tensor  # shape: (24, max_partitions, 128)`
- `silu_out: Tensor              # shape: (8192,)`
- `logits: Tensor                # shape: (128256,)`

Block sizes:
- `up_gate_proj_block_size: 16`
- `down_proj_block_size: 16`
- `o_proj_block_size: 16`
- `lm_head_block_size: 32`
- `matvec_reduction_size: 1024`  (divides both 8192 and 3072)
- `qkv_block_size: 16`
- `attn_kv_block_size: 16`
- `attn_reduction_size: 3`   (GQA_RATIO = 3)

The instruction opcodes (1–7) remain identical since the kernels define the same op types.

### Step 13: Create `megakernels/llama_3b/scheduler.py`  ✅ DONE

Adapt `make_globals()`:
- All buffer shapes use 3B dimensions
- `barriers` shape: `(28, 10, 24 + 8*2)` = `(28, 10, 40)`
- `matvec_reduction_size = 1024` (to handle 8192 and 3072 evenly)

Adapt `make_dag()` and `make_dag_layer()`:
- `schedule_qkv`: QKV out-dim = `(24+16)×128 = 5120`, num blocks = `5120/16 = 320`
- `schedule_upgate`: `8192/16 = 512` blocks (same as 1B)
- `schedule_downproj`: `num_col_splits = 8192 / 1024 = 8`, `num_down_blocks = 3072/16 = 192`
- `schedule_lm_head`: `128256/32 = 4008` blocks

### Step 14: Create `megakernels/llama_3b/python_vm.py`  ✅ DONE

Same structure as the 1B `python_vm.py`. All barrier counts are derived from globals dimensions rather than hard-coded:
- `o_proj_residual`: barrier check expects `num_attention_heads = 24`
- `down_proj_residual`: barrier check expects `intermediate_size/up_gate_proj_block_size = 8192/16 = 512`
- `layer_norm_double_matvec_silu`: barrier check expects `hidden_size/o_proj_block_size = 3072/16 = 192`
- `layer_norm_matvec_rope_append`: barrier for previous layer's down_proj expects `(intermediate_size/matvec_reduction_size) × (hidden_size/down_proj_block_size) = 8×192 = 1536`
- `partial_attention`: barrier checks use `blocks_per_head = head_dim/qkv_block_size = 128/16 = 8` (was 4)
- `rms_lm_head`: barrier expects down_proj completion on layer 27

### Step 15: Create `megakernels/llama_3b/mk.py`  ✅ DONE

```python
class Latency3B_MK_Interpreter(MK_Interpreter):
    def __init__(self, mk_dir: Path):
        self.mk_func = get_mk_3b_func(mk_dir)

    def interpret(self, globs: Globals):
        interpret_with_mk(globs, self.mk_func)
```

The `interpret_with_mk` function passes the same set of tensor arguments to the pybind module (since the globals struct has the same fields, just different sizes).

Loads `mk_llama_3b` module name from the 3B CUDA build directory.

### Step 16: Update `dispatch.py`  ✅ DONE

Added the 3B latency setting:

```python
from megakernels.llama_3b.mk import Latency3B_MK_Interpreter
from megakernels.llama_3b.python_vm import INSTRUCTION_TO_SOLVER as LATENCY_3B_INSTRUCTION_TO_SOLVER
from megakernels.llama_3b.scheduler import Latency3B_ScheduleBuilder

BUILDER_MAP["latency_3b"] = Latency3B_ScheduleBuilder
MK_INTERPRETER_MAP["latency_3b"] = Latency3B_MK_Interpreter
INSTRUCTION_TO_SOLVER_MAP["latency_3b"] = LATENCY_3B_INSTRUCTION_TO_SOLVER
```

### Step 17: Update scripts  ✅ DONE

In `benchmark.py` and `generate.py`, added convenience methods:
```python
def l3(self):
    self.model = "meta-llama/Llama-3.2-3B-Instruct"
    self.setting = "latency_3b"
    self.mk_dir = Path(__file__).parent.parent.parent / "demos" / "low-latency-llama-3b"
```

---

## 4. Critical Technical Challenges

### 4.1 Shared Memory Budget

With `hidden_dim=3072`, the activation vector is 6KB (bf16). Key constraints:

| Component | 1B Size | 3B Size | Notes |
|-----------|---------|---------|-------|
| Activation vector (`sv_bf<hidden_dim>`) | 4KB | 6KB | Loaded into shared memory for RMS norm + matvec |
| RMS scale vector (`sv_bf<hidden_dim>`) | 4KB | 6KB | Layer norm weights |
| Weight tile (`st_bf<16, 512>`) | 16KB | 16KB | Same tile size, but need more tiles per iteration |
| Per-matvec scratch (16 warps × 64B) | 1KB | 1KB | Partial sums for reduction |
| Total static shared | ~4.5KB | ~4.5KB | Config scratch, semaphores, etc. |
| Dynamic shared (pages) | ~208KB | ~208KB | 13 pages × 16KB |

The matvec pipeline needs activation (1 page) + weight stages. With 6 pages per weight stage (to cover 3072 cols), and 1 activation page:
- 1 stage: 1 + 6 = 7 pages ✓ (6 spare)
- 2 stages: 1 + 12 = 13 pages ✓ (exactly fits!)
- 3 stages: 1 + 18 = 19 pages ✗
- **Solution**: Use 2 pipeline stages for good overlap. Falls back to 1 if needed.

### 4.2 Down-Projection Column Splits (8192 not divisible by 3072)

The `schedule_downproj` function uses `num_col_splits = intermediate_size // matvec_reduction_size`. For 3B: `8192 / 3072 ≈ 2.67` — not an integer.

**Solution**: Set `matvec_reduction_size = 1024`. Then:
- `num_col_splits = 8192 / 1024 = 8` for down-projection
- Each SM processes a 16-row × 1024-col slice
- `1024` also divides `hidden_size = 3072` evenly (`3072 / 1024 = 3`)
- The CUDA `MatVecAddOp` must use `reduction_block_idx * reduction_size` as start column (parameterized from globals, not from `hidden_dim`)

### 4.3 RoPE with head_dim=128

The 1B RoPE implementation processes 64-element heads:
- Each matvec block of 16 elements covers 16/64 = 1/4 of a head
- Rope cos/sin scratch: 256 bytes (64 floats)
- `__shfl_sync` with `laneid() < 16` processes 16 pairs = 32 values (half-head interleaved)

For 128-element heads:
- Each block of 16 covers 16/128 = 1/8 of a head
- Need 512 bytes for cos, 512 for sin = 1024 total rope scratch
- Verify scratch budget: `SCRATCH_BYTES = 4096`, currently using last 512 bytes for rope. We'd use last 1024 bytes — should fit.
- `head_chunk = block_idx % 8` (was `% 4`)
- The `__shfl_sync` pair approach still works since we process 16 elements at a time (8 interleaved pairs), but the rope lookup offset changes.

### 4.4 GQA Ratio Change (4 → 3)

The GQA ratio changes from 4 (1B: 32 Q heads / 8 KV heads) to 3 (3B: 24 Q heads / 8 KV heads). This affects:

- **`attention_partial.cu`**: The `store_4_rows` / per-warp head logic must handle grouping 3 Q heads per KV head. The `Q_HEADS_PER_INSTRUCTION` likely needs to change from 4 to 3.
- **`attention_reduction.cu`**: `attn_reduction_size` should be 3 (1 KV head's worth of Q heads at a time).
- **Q loading**: Load 3 heads × 128 dim = 384 elements = 768 bytes per KV head group (was 4 × 64 = 256 elements = 512 bytes).
- **O-proj arrival count**: 24 Q heads complete attention (was 32).

### 4.5 Attention Tile Sizes

With `head_dim=128`, the attention MMA is `Q(3×128) @ K.T(128×16) = scores(3×16)`:
- `rt_bf<16, 128>` register tile for Q — requires 2 tiles per row in the 16-wide MMA format
- The flash-attention loop is unchanged structurally (iterate over KV blocks), but each step loads larger K and V tiles
- Output accumulator grows: `rt_fl<16, 128>` = 2× registers

**Register pressure**: The 1B attention consumer uses ~20 register tiles. With head_dim doubled, expect ~30+. Verify this fits in the 104 consumer registers budget (`CONSUMER_REGISTERS = 104`). If not, may need to increase or reduce other register usage.

### 4.6 Memory Bandwidth Considerations (3B vs 1B)

Llama 3B is ~2.5× larger than 1B in parameter count (3.2B vs 1.2B parameters). For single-token decode, performance is memory-bandwidth-bound:

- **Total weight reads per token (1B)**: ~2.4 GB (1.2B params × 2 bytes)
- **Total weight reads per token (3B)**: ~6.4 GB (3.2B params × 2 bytes)
- **H100 HBM bandwidth**: ~3.35 TB/s → theoretical min latency: ~1.9ms/token for 3B (vs ~0.7ms for 1B)
- **B200 HBM bandwidth**: ~8 TB/s → theoretical min latency: ~0.8ms/token for 3B

The megakernel approach aims to maximize HBM utilization by overlapping loads across SMs. With 132 SMs (H100), each SM handles `3.2B_params / 132 ≈ 24M params` worth of work per token.

### 4.7 Weight Memory

Llama 3B in bf16 requires ~6.4 GB for weights alone. KV cache at max_len=16384 with 8 heads × 128 dim × 28 layers × 2 (K+V) = ~0.9 GB. Total ~7.3 GB — fits easily on a single H100 80GB.

---

## 5. Implementation Order

1. **Python side first** (can test with `pyvm` mode without compiling CUDA):  ✅ DONE
   1. `megakernels/llama_3b/__init__.py`
   2. `megakernels/llama_3b/instructions.py`
   3. `megakernels/llama_3b/scheduler.py`
   4. `megakernels/llama_3b/python_vm.py`
   5. `megakernels/llama_3b/mk.py`
   6. Update `dispatch.py`
   7. Verify with `python megakernels/scripts/generate.py mode=pyvm setting=latency_3b .l3`

2. **CUDA side** (requires compilation):
   1. `demos/low-latency-llama-3b/llama_3b.cuh` — globals, constants
   2. `demos/low-latency-llama-3b/utils.cuh` — parameterize rms_norm
   3. `demos/low-latency-llama-3b/matvec_pipeline.cuh` — update for 3072 hidden dim
   4. `demos/low-latency-llama-3b/rms_matvec_rope_append.cu` — QKV op with 128-dim heads
   5. `demos/low-latency-llama-3b/attention_partial.cu` — attention tiles for 128-dim, GQA=3
   6. `demos/low-latency-llama-3b/attention_reduction.cu` — reduction for 128-dim, GQA=3
   7. `demos/low-latency-llama-3b/matvec_adds.cu` — o_proj + downproj
   8. `demos/low-latency-llama-3b/upgate.cu` — up/gate with 8192 intermediate
   9. `demos/low-latency-llama-3b/rms_lm_head.cu` — final LM head
   10. `demos/low-latency-llama-3b/llama_3b.cu` — entry point + pybind
   11. `demos/low-latency-llama-3b/Makefile`

3. **Testing**:
   1. Run `mode=pyvm` to validate the Python reference against `mode=torch`
   2. Compile CUDA kernel, run `mode=mk`
   3. Validate output tokens match the torch reference
   4. Benchmark: `python megakernels/scripts/benchmark.py mode=mk setting=latency_3b .l3`

---

## 6. Risk Summary

| Risk | Severity | Mitigation |
|------|----------|------------|
| Shared memory for matvec pipeline with 3072 hidden dim | Medium | 6 pages/stage; 2 stages fit exactly in 13 pages |
| 8192 intermediate dim not divisible by 3072 hidden dim | High | Use `matvec_reduction_size=1024` with 8 col splits |
| GQA ratio change (4→3) in attention kernels | Medium | Update Q loading, store helpers, and head grouping logic |
| Attention register pressure with head_dim=128 | Medium | Profile PTXAS register usage; may need to reduce consumer warps or spill |
| RoPE scratch memory for 128-dim heads | Low | 1024 bytes of 4096 scratch; fits easily |
| KV tiles doubled → attention shared memory tight | Medium | Reduce attention pipeline stages from 3 to 2 |
| Compilation time increase | Low | Modular compilation; only rebuild changed .cu files |
