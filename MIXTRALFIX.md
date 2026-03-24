# Mixtral MK Router Failure Investigation (No Fix Applied)

## Scope
This document records the current failure point, investigation data, and a verification-first plan to fix the issue without introducing regressions.

## Current Failure Point
Reproduced with:

- `uv run python demos/mixtral/debug_mixtral_mk.py`

Observed stage status:

- `qkv`: PASS
- `partial`: PASS
- `reduction`: PASS (skipped because `num_partitions=1`)
- `oproj`: PASS
- `router`: FAIL (first failing stage)

Key failing router outputs:

- `selected_expert_indices`: `py=[1, 2]`, `mk=[0, 1]`
- `selected_expert_scores`: `py=[1.0, 0.000085]`, `mk=[0.5, 0.5]`
- `router_normed_hidden`: large mismatch; MK values near/all zero in baseline run

## New Debug Scripts Added

- `demos/mixtral/debug_router_issue.py`
- `demos/mixtral/debug_router_ablation.py`

Both were run and their outputs are summarized below.

## Evidence Collected

### 1) Pre-router state is mostly sane
From `debug_router_issue.py`:

- Pre-router (`stop_after_op='oproj'`) hidden-state parity is acceptable for current tolerance:
  - `hidden_states py-vs-mk: max_adiff=0.500000, mean_adiff=0.068886, mean_rdiff=0.011899`

Interpretation:

- Upstream stages are not catastrophically broken.
- Router failure is likely localized to router opcode implementation/runtime behavior.

### 2) Router outputs are inconsistent with MK pre-router inputs
From `debug_router_issue.py`:

- Router stage divergence:
  - `selected_expert_indices: py=[1, 2], mk=[0, 1]`
  - `selected_expert_scores: py=[1.0, 8.5e-05], mk=[0.5, 0.5]`
  - `router_normed_hidden py-vs-mk: max_adiff=4.312500, mean_adiff=0.626172, mean_rdiff=1.999838`

- MK pre-router input sanity:
  - `pre_hidden absmax=115.500000, l2=796.988464`
  - `ffn_ln_weight absmax=3.531250, l2=21.827024`

- Reference math using MK pre-router inputs predicts PyVM-like routing (not MK-like):
  - `ref_topk_from_mk_inputs: idx=[1, 2], scores=[0.999922, 7.8e-05]`
  - MK still returns `idx=[0, 1], scores=[0.5, 0.5]`

Interpretation:

- Inputs to router are nonzero and should produce strongly non-uniform routing.
- MK router behavior is inconsistent with mathematically expected result from its own inputs.

### 3) Router instruction encoding is likely not the issue
From `debug_router_issue.py`:

- Exactly one router instruction is present in host schedule and tensorized instruction buffer.
- Instruction words match expected payload:
  - host: `[5, 0]`
  - tensor: `[5, 0, 0, 0, 0, 0]`

Interpretation:

- Scheduler/tensorization for opcode/layer appears correct.
- Root cause is more likely in router kernel compute/load/store path.

### 4) Controlled ablation confirms router norm output path is broken
From `debug_router_ablation.py` (simplified deterministic setup):

- With deterministic nonzero hidden input and simple weights:
  - `router_normed_hidden: py_absmax=1.726562, mk_absmax=0.000000`
  - indices diverge (`py=[1,0]`, `mk=[0,1]`)

Interpretation:

- Even in a controlled setup, MK router norm output collapses to zero.
- This strongly points to router normalization output not being correctly produced/consumed.

### 5) Sentinel prefill test indicates unstable/incorrect router buffer behavior
From `debug_router_ablation.py` with `mk_prefill=1.0` on `router_normed_hidden` before launch:

- Post-run `router_normed_hidden` becomes `NaN` (`mk_absmax=nan`, first entries `nan`)
- `selected_expert_scores` become `nan`

Interpretation:

- Router path likely has incorrect memory semantics or wrong API usage for writing/reading `router_normed_hidden`.
- Behavior is not just a deterministic numeric drift; it can produce invalid values based on buffer state.

## Most Likely Root Causes (Ranked)

1. **Router global write/read path for `router_normed_hidden` is incorrect**
   - In `demos/mixtral/router.cu`, router computes normed chunks and then writes to global before warp-0 reads back for logits.
   - The symptom pattern (zeros in baseline, NaNs with sentinel prefill) is consistent with an invalid store/load path or wrong API usage/overload for global memory write.

2. **Memory ordering/visibility bug between multi-warp write and warp-0 read of `router_normed_hidden`**
   - If writes are not properly synchronized/fenced for global visibility, warp-0 may read stale or undefined data.
   - Current code uses `group::sync`, but this may be insufficient depending on how the specific store primitive behaves.

3. **Router norm compute call contract mismatch (less likely than #1/#2)**
   - If helper usage (arguments/layout/scratch) is semantically wrong, output could be corrupted.
   - However, deterministic zero + NaN sensitivity suggests store/load semantics are a stronger candidate.

## Best Verification Path (Before Any Fix)

1. **Instrument router kernel with a minimal debug mode**
   - Add temporary debug outputs for:
     - one warp chunk of computed `act_vec` (pre-global-store)
     - first few elements actually read by warp-0 for dot-products
   - If `act_vec` is finite/nonzero but readback is zero/NaN, issue is confirmed in global write/read path.

2. **A/B test read source in router kernel**
   - Temporary experiment: compute logits directly from the in-register/SMEM normed vector path instead of re-reading `router_normed_hidden` from GMEM.
   - If this restores top-k parity, global buffer path is conclusively the problem.

3. **A/B test store primitive**
   - Replace current `kittens::warp::store(...router_normed_hidden...)` path with the store pattern used by other stable kernels in this repo (TMA or known-good global-store helper pattern).
   - Re-run `debug_router_issue.py` and `debug_router_ablation.py`.

4. **Add finite checks in debug build**
   - Assert/record finiteness of `router_normed_hidden` and `selected_expert_scores` in debug mode to catch corruption early.

## Best Fix Path (After Verification)

1. **Fix router normed-hidden producer/consumer path**
   - Ensure a known-good global store operation is used for `router_normed_hidden`.
   - Ensure proper synchronization/fence before warp-0 global readback.

2. **Optionally remove fragile GMEM round-trip inside router**
   - Compute router logits from already available normed chunks (SMEM/register reduction), then store `router_normed_hidden` for downstream experts.
   - This can reduce the chance of intra-kernel visibility bugs.

3. **Regression validation**
   - Run:
     - `uv run python demos/mixtral/debug_mixtral_mk.py`
     - `uv run python demos/mixtral/debug_mixtral_mk.py --no-stop`
     - `uv run python demos/mixtral/debug_router_issue.py`
     - `uv run python demos/mixtral/debug_router_ablation.py`
   - Confirm router parity and that downstream stages improve accordingly.

## Notes
- No production kernel fix was applied during this investigation.
- This report is evidence-first and intended to de-risk implementation by narrowing the fault to a specific kernel path.
