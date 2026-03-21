# Mixtral 7Bx8 Megakernel — Implementation Plan

## 1. Model Overview and Key Differences from Llama

Mixtral 8x7B is a Sparse Mixture of Experts (SMoE) model. Each transformer block has:
- **Attention**: Identical to Llama with GQA (32 heads, 8 kv heads, head_dim=128)
- **MoE MLP**: A learned router that selects 2 out of 8 expert MLPs per token

Key architecture constants:
```
hidden_size          = 4096
num_hidden_layers    = 32
num_attention_heads  = 32
num_kv_heads         = 8
head_dim             = 128
intermediate_size    = 14336   # per expert (vs 8192 for Llama-3.2-3B)
num_experts          = 8
num_experts_per_tok  = 2
vocab_size           = 32000
```

Weight sizes at bfloat16:
- Per-layer per-expert gate/up proj: [14336 × 4096] = ~115MB × 2 × 8 = 1.84GB/layer
- Per-layer per-expert down proj: [4096 × 14336] = ~115MB × 8 = ~920MB/layer
- Total per layer (MoE only): ~2.76GB; total model: ~88GB → must run on multi-GPU or quantized

### The Core MoE Challenge for Megakernels

The critical design tension: **expert selection is data-dependent at runtime**, but instruction schedules must be compiled before the token is seen. Options:

**Option A — Schedule all 8 experts, CUDA skips inactive (recommended)**
Each token, 2 out of 8 experts run. Schedule all 8 UpGate + DownProj instructions unconditionally; the CUDA kernel reads `expert_indices` from globals and returns early (NoOp) if the expert is not active. This wastes 6 SM executions per layer, but each skip is a fast branch — the bottleneck is the 2 active experts' HBM bandwidth.

Pros: static schedule, no rebuild, trivially correct, same scheduling framework
Cons: 6/8 instructions per layer are NoOps (idle SM cycles)

**Option B — Rebuild schedule each token (not recommended for initial implementation)**
After the router runs in Python, re-assign only 2 experts and re-tensorize. High overhead (~ms per token) and breaks the pipelined kernel model.

**Recommendation**: Start with Option A. A future optimization could speculatively issue only the 2 cheapest expected experts or pre-tile expert weight loads.

---

## 2. Python Layer

### 2.1 Model Implementation: `megakernels/mixtral.py`

Mirror `megakernels/llama.py` but for Mixtral. Key differences:

**Weight loading from HuggingFace** (`mistralai/Mixtral-8x7B-Instruct-v0.1`):
- HF safetensors use keys like `model.layers.{i}.block_sparse_moe.experts.{j}.w1.weight` (gate)
- `w2` = down proj, `w3` = up proj
- Router: `model.layers.{i}.block_sparse_moe.gate.weight` — shape [num_experts, hidden_size]

**StackedParams for Mixtral**:
```python
@dataclass
class MixtralStackedParams:
    # Attention (identical to Llama)
    qkv_proj:       Tensor  # [num_layers, num_q+2*num_kv heads * head_dim, hidden_size]
    o_proj:         Tensor  # [num_layers, hidden_size, hidden_size]
    attn_ln_weight: Tensor  # [num_layers, hidden_size]
    # MoE
    router_weight:  Tensor  # [num_layers, num_experts, hidden_size]
    expert_gate:    Tensor  # [num_layers, num_experts, intermediate_size, hidden_size]
    expert_up:      Tensor  # [num_layers, num_experts, intermediate_size, hidden_size]
    expert_down:    Tensor  # [num_layers, num_experts, hidden_size, intermediate_size]
    ffn_ln_weight:  Tensor  # [num_layers, hidden_size]  (pre-MoE layernorm)
    # LM head
    lm_head_norm_weight: Tensor  # [hidden_size]
    lm_head_weight:      Tensor  # [vocab_size, hidden_size]
```

**setup_caches**: Identical pattern to Llama; stacked KV cache shape `[2, num_layers, num_kv_heads, max_seq_len, head_dim]`.

**PyTorch forward pass**: Implement a reference `MixtralForCausalLM.forward()` for correctness comparison. The MoE forward:
1. Compute router logits: `hidden @ router_weight.T` → [num_experts]
2. Softmax → top-2 selection → expert_indices [2], expert_weights [2] (normalized)
3. For each selected expert: gate_up_silu + weighted down proj, accumulate result

### 2.2 Globals and Instructions: `megakernels/mixtral/instructions.py`

**Globals**:
```python
@dataclass
class MixtralGlobals(BaseGlobals):
    # --- attention params (same as latency Globals) ---
    post_ln_rope_q:            Tensor
    attn_out:                  Tensor
    attn_lse_intermediates:    Tensor
    attn_out_intermediates:    Tensor

    # --- MoE params ---
    expert_silu_out:           Tensor  # [num_experts, intermediate_size] — scratch per expert
    logits:                    Tensor  # [vocab_size]
    # runtime router output
    selected_expert_indices:   Tensor  # [num_experts_per_tok] int32 — filled by Router instruction
    selected_expert_weights:   Tensor  # [num_experts_per_tok] float — filled by Router instruction

    # stacked weights (new vs Llama)
    router_weights:  Tensor  # [num_layers, num_experts, hidden_size]
    expert_gate:     Tensor  # [num_layers, num_experts, intermediate_size, hidden_size]
    expert_up:       Tensor  # [num_layers, num_experts, intermediate_size, hidden_size]
    expert_down:     Tensor  # [num_layers, num_experts, hidden_size, intermediate_size]
    ffn_ln_weights:  Tensor  # [num_layers, hidden_size]

    # constants
    num_experts:         int
    num_experts_per_tok: int
    # block sizes
    expert_proj_block_size: int
    ...
```

**Instruction Opcodes**:

| # | Name | Description | New? |
|---|------|-------------|------|
| 1 | `LayerNorm_QKV_MatVecRopeAppend` | Same as Llama latency | No |
| 2 | `PartialAttention` | Same | No |
| 3 | `AttentionReduction` | Same | No |
| 4 | `O_ProjResidual` | Same | No |
| 5 | `MoE_Router` | RMS norm + router matVec + softmax + top-2 | **Yes** |
| 6 | `ExpertUpGateSiLU` | Gate+Up matVec+SiLU for one expert | **Yes** |
| 7 | `ExpertDownProjAccum` | Down proj + weighted accumulate into hidden | **Yes** |
| 8 | `RMS_LM_Head` | Same as Llama | No |

**MoE_Router** (opcode 5):
```python
@dataclass
class MoE_Router(Instruction):
    layer_idx: int
    # single instruction — runs RMS norm + matVec [num_experts, hidden] + top-2 selection
    # writes selected_expert_indices and selected_expert_weights to globals
    @classmethod
    def opcode(cls): return 5
    @classmethod
    def prev_opcode(cls): return O_ProjResidual.opcode()
    def cost(self, globs): return globs.num_experts * globs.hidden_size
```

**ExpertUpGateSiLU** (opcode 6):
```python
@dataclass
class ExpertUpGateSiLU(Instruction):
    layer_idx: int
    expert_idx: int
    block_idxs: list[int]  # which output blocks of intermediate_size this SM handles
    @classmethod
    def opcode(cls): return 6
    @classmethod
    def prev_opcode(cls): return MoE_Router.opcode()
    def cost(self, globs):
        # Only 2/8 experts actually run; cost reflects the skip probability
        return len(self.block_idxs) * globs.expert_proj_block_size * globs.hidden_size * 2
```

**ExpertDownProjAccum** (opcode 7):
```python
@dataclass
class ExpertDownProjAccum(Instruction):
    layer_idx: int
    expert_idx: int
    start_block_idx: int
    end_block_idx: int
    reduction_block_idx: int
    @classmethod
    def opcode(cls): return 7
    @classmethod
    def prev_opcode(cls): return ExpertUpGateSiLU.opcode()
```

**Serialization note**: All instructions serialize to 32 int32 words. `block_idxs` for `ExpertUpGateSiLU` must be encoded as `start_block_idx + num_blocks` (contiguous range), not as a variable-length list, to fit in the fixed 32-word format.

### 2.3 Scheduler: `megakernels/mixtral/scheduler.py`

DAG per layer (latency mode, batch_size=1):

```
prev_layer_outputs
        │
   ┌────▼────┐
   │  QKV    │ × sm_count  (parallel, same as Llama)
   └────┬────┘
        │
   ┌────▼────────┐
   │PartialAttn  │ × num_kv_heads × num_partitions  (parallel)
   └────┬────────┘
        │ (if num_partitions > 1)
   ┌────▼──────────┐
   │AttnReduction  │ × num_kv_heads
   └────┬──────────┘
        │
   ┌────▼──────────┐
   │ O_ProjResidual│ × num_o_blocks  (parallel)
   └────┬──────────┘
        │ all o_proj_nodes
   ┌────▼──────┐
   │ MoE_Router│  (single node — sequential bottleneck)
   └────┬──────┘
        │
   ┌────▼─────────────────────────────┐
   │ ExpertUpGateSiLU × 8 × sms_each │ (all 8 experts, CUDA skips inactive)
   └────┬─────────────────────────────┘
        │ (each ExpertDown depends on its corresponding ExpertUp for same expert)
   ┌────▼────────────────────────────────┐
   │ ExpertDownProjAccum × 8 × sms_each  │
   └────┬────────────────────────────────┘
        │ all downproj nodes
   next layer QKV
```

**SM assignment for experts**: Each expert's UpGate and DownProj split across multiple SMs (same block-splitting logic as Llama's `schedule_upgate` / `schedule_downproj`, but multiplied by `num_experts=8`).

**Barrier design**: The `barriers` tensor needs an extra dimension for expert index, or expert ops use a per-expert opcode-level barrier slot. Recommended:
- Allocate barriers `[num_layers, num_opcodes, max(num_attention_heads + 2*num_kv_heads, num_experts)]`
- Expert barrier slot: `barriers[layer, 6, expert_idx]` (opcode 6 = ExpertUpGateSiLU)
- Router barrier: `barriers[layer, 5, 0]` — signals all ExpertUpGate nodes

**Key `make_globals` additions**:
```python
selected_expert_indices = torch.zeros(num_experts_per_tok, dtype=torch.int32, device=device)
selected_expert_weights = torch.zeros(num_experts_per_tok, dtype=torch.bfloat16, device=device)
expert_silu_out = torch.zeros([num_experts, intermediate_size], dtype=dtype, device=device)
```

### 2.4 Python VM: `megakernels/mixtral/python_vm.py`

Reference CPU/GPU implementations for each new opcode:

```python
def solve_router(globs: MixtralGlobals, ins: MoE_Router):
    # 1. RMS norm on hidden_states
    normed = rms_norm(globs.hidden_states, globs.ffn_ln_weights[ins.layer_idx], globs.rms_norm_eps)
    # 2. Router matVec: [num_experts, hidden] @ [hidden] → [num_experts]
    router_logits = globs.router_weights[ins.layer_idx] @ normed
    # 3. Softmax + top-k
    scores = torch.softmax(router_logits, dim=-1)
    top_weights, top_indices = torch.topk(scores, globs.num_experts_per_tok)
    top_weights = top_weights / top_weights.sum()  # renormalize
    globs.selected_expert_indices.copy_(top_indices.int())
    globs.selected_expert_weights.copy_(top_weights)
    # save normed hidden for reuse by ExpertUpGate
    globs._router_normed = normed  # scratch, not serialized


def solve_expert_upgate(globs: MixtralGlobals, ins: ExpertUpGateSiLU):
    # Check if this expert was selected
    if ins.expert_idx not in globs.selected_expert_indices.tolist():
        return  # skip
    gate_out = globs.expert_gate[ins.layer_idx, ins.expert_idx] @ globs._router_normed
    up_out   = globs.expert_up[ins.layer_idx, ins.expert_idx]   @ globs._router_normed
    globs.expert_silu_out[ins.expert_idx] = F.silu(gate_out) * up_out


def solve_expert_downproj(globs: MixtralGlobals, ins: ExpertDownProjAccum):
    if ins.expert_idx not in globs.selected_expert_indices.tolist():
        return
    slot = (globs.selected_expert_indices == ins.expert_idx).nonzero()[0].item()
    weight = globs.selected_expert_weights[slot]
    contribution = globs.expert_down[ins.layer_idx, ins.expert_idx] @ globs.expert_silu_out[ins.expert_idx]
    globs.hidden_states[ins.start_block_idx * BS : ins.end_block_idx * BS] += weight * contribution[...]
```

**Correctness note on accumulation**: `ExpertDownProjAccum` accumulates into `hidden_states` (the residual stream) from multiple SMs and potentially 2 experts. Need atomic-style add — in PyVM, the sequential execution order guarantees this is safe if instructions are ordered correctly. In CUDA, this requires careful synchronization (see Section 3.2).

---

## 3. CUDA Kernel Layer

### 3.1 New Demo Directory: `demos/mixtral/`

Structure mirroring `demos/low-latency-llama/`:
```
demos/mixtral/
├── Makefile
├── mixtral.cu           # pybind11 entry point, globals struct, kernel instantiation
├── globals.cuh          # globals struct definition
├── router.cu            # opcode 5: RMS norm + router matvec + top-k
├── expert_upgate.cu     # opcode 6: gate+up matvec + silu (conditional on expert active)
├── expert_downproj.cu   # opcode 7: down proj + weighted atomic-add to hidden (conditional)
├── attention_partial.cu # opcode 2: reuse from llama (same logic, different tile sizes)
├── attention_reduction.cu # opcode 3: reuse
├── rms_matvec_rope_append.cu # opcode 1: reuse
├── matvec_adds.cu       # opcode 4: reuse
└── rms_lm_head.cu       # opcode 8: reuse
```

Reuse vs. rewrite: Opcodes 1-4 and 8 use identical math to the Llama latency demo. They can be copied verbatim, but **tile sizes may need adjustment** since `hidden_size=4096` (2× larger than Llama-1B's 2048). ThunderKittens tile configs in `config.cuh` will need reviewing.

### 3.2 New CUDA Ops

#### `router.cu` — `MoE_Router` op

```
Inputs (loaded via TMA):
  - hidden_states[hidden_size]              (global)
  - ffn_ln_weights[layer_idx][hidden_size]  (global)
  - router_weights[layer_idx][num_experts][hidden_size]  (global)

Compute:
  1. Consumer warps: RMS norm of hidden_states (parallel across hidden_size)
  2. Consumer warps: matVec [8, 4096] × [4096] → 8 scalars (tiny — one warp group handles all)
  3. Softmax over 8 scalars (register-level, no shared mem needed)
  4. ArgTopK-2: write indices and normalized weights to globals

Outputs (written to global):
  - selected_expert_indices[2]   int32
  - selected_expert_weights[2]   bfloat16
  - router_normed_hidden[hidden_size]  (reused by all ExpertUpGate instructions)
```

The router matVec is tiny (8×4096), so this instruction is not compute-bound. The latency cost is mostly RMS norm + HBM reads for router_weights.

Synchronization: After router completes, ALL ExpertUpGateSiLU instructions depend on it. The barrier at `barriers[layer, 5, 0]` signals completion.

#### `expert_upgate.cu` — `ExpertUpGateSiLU` op

```
Inputs:
  - router_normed_hidden[hidden_size]   (global — written by router)
  - expert_gate[layer][expert][block:block+BS][hidden_size]   (TMA load)
  - expert_up[layer][expert][block:block+BS][hidden_size]     (TMA load)
  - selected_expert_indices[2]          (read to check if active)

Conditional skip:
  At start of kernel op, read selected_expert_indices.
  If ins.expert_idx not in selected_expert_indices → return immediately.
  (This check should happen in the Controller warp before issuing loads.)

Compute (only if active):
  1. TMA load blocks of gate_proj and up_proj rows
  2. gate_out = gate_rows @ normed_hidden    (matvec, block at a time)
  3. up_out   = up_rows   @ normed_hidden
  4. silu_out[expert][block] = silu(gate_out) * up_out

Outputs:
  - expert_silu_out[expert][start_block:end_block][intermediate_dim]
```

Block structure mirrors `upgate.cu` in Llama but indexes by `expert_idx`. The intermediate_size=14336 is larger than Llama's 8192; tile sizes may need adjustment.

#### `expert_downproj.cu` — `ExpertDownProjAccum` op

```
Inputs:
  - expert_silu_out[expert][intermediate_size]   (global)
  - expert_down[layer][expert][block:block+BS][intermediate_size]  (TMA load)
  - selected_expert_indices, selected_expert_weights   (read for weight)

Conditional skip:
  Same as UpGate — check active before loading.

Compute (only if active):
  1. Look up weight for this expert from selected_expert_weights
  2. TMA load rows of down_proj
  3. output_partial = down_rows @ expert_silu_out[expert]  (matvec)
  4. Atomic add: hidden_states[block] += weight * output_partial

Outputs:
  - hidden_states (atomic add, no lock needed if reduction_block_idx partitions correctly)
```

**Accumulation strategy**: Two experts contribute to `hidden_states`. Since each DownProj instruction is assigned a distinct range of `hidden_states` output blocks (`start_block_idx:end_block_idx`), and each expert independently writes its weighted contribution, atomic adds are safe as long as no two instructions for the same `(layer, block_range)` run concurrently. This is guaranteed by the DAG (all ExpertDownProj for a layer write different block ranges, and the barrier between layers enforces ordering).

However, two experts writing to the **same output block** at the **same time** on different SMs is a race. The safest fix: serialize ExpertDownProj instructions for the same block range across experts (add inter-expert dependency in the DAG), or use separate accumulation buffers and a final reduce. The simplest solution for initial correctness: run all expert 0's DownProj first, then all expert 1's.

### 3.3 Globals Struct in CUDA (`demos/mixtral/globals.cuh`)

Extends the Llama globals struct with:
```cpp
// MoE
bf16 *router_weights;         // [num_layers, num_experts, hidden_size]
bf16 *expert_gate;            // [num_layers, num_experts, intermediate_size, hidden_size]
bf16 *expert_up;              // [num_layers, num_experts, intermediate_size, hidden_size]
bf16 *expert_down;            // [num_layers, num_experts, hidden_size, intermediate_size]
bf16 *ffn_ln_weights;         // [num_layers, hidden_size]
bf16 *router_normed_hidden;   // [hidden_size]
bf16 *expert_silu_out;        // [num_experts, intermediate_size]
int  *selected_expert_indices;// [num_experts_per_tok]
float *selected_expert_weights;// [num_experts_per_tok]
int   num_experts;
int   num_experts_per_tok;
```

### 3.4 ThunderKittens Tile Configurations (`config.cuh`)

For Mixtral (hidden_size=4096, intermediate_size=14336):
- QKV output dim: (32 + 2×8) × 128 = 6144 — larger than Llama-1B (2048)
- UpGate output dim: 14336 — larger than Llama-1B (8192)
- DownProj input dim: 14336

Review and possibly increase:
- Page size (currently 16KB); may need 32KB pages for wider tiles
- `matvec_reduction_size`: `gcd(14336, 4096)` = 512
- Block sizes: start with same 16-row blocks, verify tile fits in shared memory

### 3.5 Kernel Registration in `mixtral.cu`

```cpp
#include "megakernel.cuh"
#include "globals.cuh"
// include each op:
#include "rms_matvec_rope_append.cu"
#include "attention_partial.cu"
#include "attention_reduction.cu"
#include "matvec_adds.cu"
#include "router.cu"
#include "expert_upgate.cu"
#include "expert_downproj.cu"
#include "rms_lm_head.cu"

using MixtralKernel = mk<MixtralConfig, MixtralGlobals,
    QKVOp, PartialAttnOp, AttnReduceOp, OProjOp,
    RouterOp, ExpertUpGateOp, ExpertDownProjOp, LMHeadOp>;
```

---

## 4. Testing Strategy

### Phase 1: Python reference implementation
**Goal**: Correct `MixtralForCausalLM` with `stack_params()` and `setup_caches()`.

```bash
python megakernels/scripts/generate.py mode=torch setting=mixtral model=mistralai/Mixtral-8x7B-Instruct-v0.1
```

Compare logits against HuggingFace `AutoModelForCausalLM` directly. Once outputs match, the stacked weight layout is confirmed correct.

### Phase 2: PyVM solver correctness
**Goal**: Each new instruction's Python solver produces the same output as the reference forward.

Add a `diff_test.py`-style script at `megakernels/scripts/diff_test_mixtral.py`:
```bash
python megakernels/scripts/diff_test_mixtral.py setting=mixtral_latency layer_limit=1 stop_after_op=router
python megakernels/scripts/diff_test_mixtral.py setting=mixtral_latency layer_limit=1 stop_after_op=expert_upgate
python megakernels/scripts/diff_test_mixtral.py setting=mixtral_latency layer_limit=1
python megakernels/scripts/diff_test_mixtral.py setting=mixtral_latency layer_limit=None  # full model
```

The existing `diff_test.py` uses `stop_after_op` extensively — replicate this for each new opcode.

**Debugging expert selection**: When comparing PyVM vs. PyTorch, force a fixed `expert_indices` selection (e.g., experts 0 and 1) to isolate accumulation bugs from router selection bugs.

### Phase 3: Instruction serialization round-trip test
**Goal**: Every instruction serializes and deserializes without data loss.

Write a unit test in `megakernels/mixtral/test_serialization.py`:
```python
for ins in [MoE_Router(0), ExpertUpGateSiLU(0, 3, [0,1,2]), ExpertDownProjAccum(0, 5, 0, 2, 0)]:
    words = ins.serialize()
    assert len(words) == 32
    assert words[0] == ins.opcode()
    reconstructed = ins.__class__.deserialize(words)
    assert reconstructed == ins
```

### Phase 4: CUDA kernel, one op at a time

Build the kernel incrementally using `stop_after_op`:

```bash
# Step 1: Test attention is correct (reuses Llama ops)
python megakernels/scripts/diff_test_mixtral.py stop_after_op=oproj layer_limit=1

# Step 2: Test router op
python megakernels/scripts/diff_test_mixtral.py stop_after_op=router layer_limit=1

# Step 3: Test expert UpGate (with fixed expert selection, force experts 0+1 active)
python megakernels/scripts/diff_test_mixtral.py stop_after_op=expert_upgate layer_limit=1 force_experts=0,1

# Step 4: Test expert DownProj + accumulation
python megakernels/scripts/diff_test_mixtral.py stop_after_op=downproj layer_limit=1

# Step 5: Full single layer
python megakernels/scripts/diff_test_mixtral.py layer_limit=1

# Step 6: Full model (requires multi-GPU or quantized weights)
python megakernels/scripts/diff_test_mixtral.py layer_limit=None
```

For each failing step, use `gpy.diff(gmk)` (already in `BaseGlobals`) to identify the first diverging tensor.

### Phase 5: End-to-end generation

```bash
python megakernels/scripts/generate.py mode=pyvm setting=mixtral_latency model=mistralai/Mixtral-8x7B-Instruct-v0.1 prompt="Tell me a joke" ntok=50
python megakernels/scripts/generate.py mode=mk   setting=mixtral_latency model=mistralai/Mixtral-8x7B-Instruct-v0.1 prompt="Tell me a joke" ntok=50
```

Compare generated text (not just logits) to HF reference.

---

## 5. Evaluation Plan

### Correctness Metrics
- **Per-instruction max abs error** vs. PyVM reference (use `gpy.diff(gmk)`)
- **Per-token logit max abs error** vs. PyTorch reference
- **Text quality**: Same greedy decoding outputs as HF reference

### Performance Benchmarks (`benchmark.py`)

Add `setting=mixtral_latency` to the dispatch map and run:
```bash
python megakernels/scripts/benchmark.py mode=mk setting=mixtral_latency ntok=200 warmup=50
```

Key metrics:
- **Time per token (ms)** — primary latency metric
- **Expert utilization**: fraction of scheduled expert instructions that actually ran (vs. skipped). Should be ~25% (2/8).
- **SM occupancy**: timings tensor shows per-SM wall time; use to identify imbalance

### Comparison Baselines
1. **PyTorch** (`mode=torch`): standard HF transformers forward
2. **vLLM** or **llama.cpp**: best-of-class latency for Mixtral (external)
3. **`mode=pyvm`**: validates scheduling overhead

### Profiling
```bash
python megakernels/scripts/make_torch_profile.py setting=mixtral_latency
```
Use NVIDIA Nsight Compute for kernel-level analysis:
- Roofline position of each expert op (are we memory-bound as expected?)
- HBM bandwidth utilization for expert weight loads
- Warp divergence due to conditional skips

---

## 6. Implementation Order

1. **`megakernels/mixtral.py`** — model weights, stacked params, PyTorch reference forward
2. **`megakernels/mixtral/instructions.py`** — Globals dataclass + 3 new instruction classes
3. **`megakernels/mixtral/python_vm.py`** — solver functions for router, expert_upgate, expert_downproj
4. **`megakernels/mixtral/scheduler.py`** — `MixtralScheduleBuilder`, `make_globals`, `make_dag`
5. **Dispatch wiring** — add `"mixtral_latency"` to `BUILDER_MAP`, `MK_INTERPRETER_MAP`, `INSTRUCTION_TO_SOLVER_MAP` in `dispatch.py`
6. **Correctness test (PyVM vs. PyTorch)** — run diff_test before touching CUDA
7. **`demos/mixtral/`** — CUDA ops, globals struct, kernel registration, Makefile
8. **CUDA correctness** — iterative diff_test per op
9. **Performance tuning** — tile sizes, prefetching, skip optimization

---

## 7. Open Questions / Future Work

- **Weight quantization**: At bfloat16, Mixtral 8x7B is ~90GB. For single-GPU testing, INT4/INT8 quantization (GPTQ or AWQ) is needed. The megakernel framework currently assumes bfloat16; quantized weights require dequantization ops or quantized matVec kernels.
- **Expert weight prefetching**: The 2 active expert weights are only known after the router runs. A speculative prefetch based on historical expert popularity could hide HBM latency.
- **Tensor parallelism**: Mixtral HF uses TP across the expert dimension (each GPU holds all 8 experts for a subset of layers, or each GPU holds a subset of experts). The existing TP sharding in `llama.py` should be extended.
- **Multi-step attention reduction for long context**: Mixtral is commonly used with 32K context (via rope_scaling). `pick_num_attention_partitions` already handles this, but CUDA partition size limits may need review.
- **Throughput demo**: A batch-mode Mixtral megakernel is more complex because different tokens in the batch may select different experts (expert token-parallelism). This is a future phase.
