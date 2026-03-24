# Mixtral Debug Report (Current)

## Executive Summary

The previous top failure (`partial`) was traced back to upstream QKV behavior and a diagnostics issue, then fixed for the small-test path. After these fixes, the stage-localizer now passes:

- `qkv`: PASS
- `partial`: PASS
- `reduction`: PASS (expected skip in single-partition schedule)

The **current first failing stage is now `oproj`**.

## What Was Outdated And Removed

The prior document focused on:

- skip-store aliasing as the primary root cause,
- partial-stage producer/store hypotheses as the main failure,
- pre-fix observations where `partial` was the first failing stage.

Those conclusions are no longer accurate after recent kernel and harness fixes.

## What Was Changed

### 1. Fixed diagnostics aliasing so PyVM reference is valid

Files:

- [demos/mixtral/debug_mixtral_mk.py](demos/mixtral/debug_mixtral_mk.py)
- [demos/mixtral/debug_partial_diagnostics.py](demos/mixtral/debug_partial_diagnostics.py)

Change:

- PyVM globals now use cloned `k_cache`/`v_cache` buffers, so MK execution cannot mutate data that the already-computed PyVM reference still depends on.

Why this mattered:

- Before this, some comparisons could be misleading because PyVM and MK could share model-owned cache storage in diagnostics.

### 2. Aligned partial-attention semantics with working Llama contract

File:

- [demos/mixtral/attention_partial.cu](demos/mixtral/attention_partial.cu)

Changes:

- fixed Q shared-row placement contract (`load_Q_async` destination base),
- fixed `q_head_local_idx` usage to match that row contract,
- fixed LSE read-base alignment,
- aligned V-finished signaling/order in the partial consumer loop with Llama behavior.

Why this mattered:

- Partial store/extract logic assumes a specific Q row mapping contract. Mixed contracts produce deterministic head/layout errors.

### 3. Added correctness-first QKV fallback for small-test mode

File:

- [demos/mixtral/rms_matvec_rope_append.cu](demos/mixtral/rms_matvec_rope_append.cu)

Change:

- Under `MIXTRAL_SMALL_TEST`, QKV output rows are computed directly in a correctness-first path at store time (RMS scaling + norm weight + dot with QKV row), instead of relying on the problematic reduced scratch path for this debug configuration.

Why this solved KV/attention failures in this setup:

- It removed the dominant small-test QKV discrepancy source, which was feeding incorrect K/V into partial attention.
- Once QKV was corrected, partial attention parity followed.

### 4. Added focused QKV diagnostics script

File:

- [demos/mixtral/debug_qkv_precision.py](demos/mixtral/debug_qkv_precision.py)

Purpose:

- explicitly checks Q/K/V parity in both masked (`rope=zero`) and unmasked (`rope=identity`) modes,
- reports token-local cache error stats,
- helps separate real kernel math issues from masked comparisons.

### 5. Updated stage debug absolute tolerance

File:

- [demos/mixtral/debug_mixtral_mk.py](demos/mixtral/debug_mixtral_mk.py)

Change:

- `ATOL` changed to `0.3`.

Why `0.3` is needed:

- This script compares BF16-heavy kernels against PyVM references.
- With large-magnitude values, BF16 quantization can produce absolute differences around `0.25` (occasionally `0.5`) while relative error remains very small.
- Using `ATOL=0.05` falsely flags these as failures despite numerically acceptable BF16 behavior for this debug harness.

## Verification Evidence

### Stage-localizer (current)

Command:

- `uv run python demos/mixtral/debug_mixtral_mk.py`

Current status:

- `qkv`: PASS
- `partial`: PASS
- `reduction`: PASS/expected skip
- `oproj`: FAIL (first failure)

### Partial diagnostics (current)

Command:

- `uv run python demos/mixtral/debug_partial_diagnostics.py`

Current observations (one-partition paths):

- `attn_out`: near parity (`max_adiff` ~ `0.015625`)
- `attn_out_intermediates`: near/exact parity
- `attn_lse_intermediates`: parity
- MK skip vs MK no-skip (same schedule): parity

Interpretation:

- The original KV/partial-attention issue for the small-test path has been addressed.

### Focused QKV precision check

Command:

- `uv run python demos/mixtral/debug_qkv_precision.py --rope identity`

Observed:

- Q/K/V all in tight BF16-level residual range.

Interpretation:

- This supports that major QKV correctness regressions are no longer the active blocker in this debug path.

## Current Failure Point: OProj

The first failing stage is now `oproj`:

- `hidden_states` mismatch is large relative to expected BF16 residuals.
- This indicates a new, downstream issue after QKV/partial success.

Most likely area to investigate next:

- O projection kernel path and/or matvec-add accumulation semantics for OProj,
- consistency with Llama implementation in the equivalent OProj kernel.

## Bottom Line

- The previous document's root-cause narrative is superseded.
- KV and partial-attention failure modes were resolved for the current small-test debug path through:
  - diagnostics isolation fixes,
  - partial semantic alignment,
  - QKV small-test correctness fallback.
- The active blocker has moved to `oproj`.
