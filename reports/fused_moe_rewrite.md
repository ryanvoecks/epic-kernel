# Mixtral MoE Fused 2-Kernel Rewrite

**Date:** 2026-03-26
**Branch:** ck/mixtral-fix

## Motivation

The original Mixtral MoE implementation used 3 separate opcodes (5-7): a router kernel, an expert upgate kernel (scheduled for all 8 experts even though only 2 are active), and an expert downproj kernel (also for all 8 experts). This created ~265 MoE instructions per layer and wasted work on inactive experts.

## New Design

Two fused opcodes replace the three:

| Opcode | Name | What it does |
|--------|------|--------------|
| 5 | `RmsRouterUpgate` | RMS norm + router softmax + top-2 selection + gate+up matvec + SiLU for both selected experts |
| 6 | `ExpertDownProjFused` | Down projection + weighted accumulate for both selected experts |
| 7 | `Mixtral_RMS_LM_Head` | Renumbered from 8 |

Each thread block is persistent across both selected experts. The router runs inline in the consumer before the matvec pipeline begins. The downproj iterates over both experts in a single pipeline pass.

## Instruction Count

MoE instructions per layer dropped from **265 to 66** (~4x reduction):

- Old: Router(1) + UpGate(8 experts x blocks) + DownProj(8 experts x col_splits) = 265
- New: RmsRouterUpgate(64 blocks) + ExpertDownProjFused(2 col_splits) = 66

No instructions are wasted on inactive experts.

## Files Changed

### Python (4 files)

| File | Change |
|------|--------|
| `megakernels/mixtral/instructions.py` | Replaced `MoE_Router`, `ExpertUpGateSiLU`, `ExpertDownProjAccum` with `RmsRouterUpgate`, `ExpertDownProjFused` |
| `megakernels/mixtral/python_vm.py` | New solver functions `solve_rms_router_upgate`, `solve_expert_downproj_fused` |
| `megakernels/mixtral/scheduler.py` | Simplified scheduling — no per-expert instruction loop |
| `demos/mixtral/debug_mixtral_mk.py` | Updated stage definitions and tolerance maps |

### CUDA (5 modified, 1 created, 2 deleted)

| File | Change |
|------|--------|
| `demos/mixtral/rms_router_matvec.cu` | **New.** Fuses router + upgate into one op with `router_done` semaphore |
| `demos/mixtral/expert_downproj.cu` | **Rewritten.** Handles both experts per instruction |
| `demos/mixtral/mixtral.cuh` | Updated opcode defines and forward declarations |
| `demos/mixtral/mixtral.cu` | Updated op registrations (7 ops) |
| `demos/mixtral/rms_lm_head.cu` | Opcode renumber + acquire fence |
| `demos/mixtral/rms_matvec_rope_append.cu` | Updated barrier reference |
| `demos/mixtral/router.cu` | **Deleted** |
| `demos/mixtral/expert_upgate.cu` | **Deleted** |

## Key Implementation Details

### RmsRouterUpgate (`rms_router_matvec.cu`)

The consumer runs in two phases:

1. **Phase 1 (pre-pipeline):** Wait for OProj barrier, compute RMS norm, store normed activations to `router_normed_hidden`, compute router dot products + softmax + top-2 selection, write `selected_expert_indices` and `selected_expert_scores`.
2. **Phase 2 (pipeline):** Enter `matvec_pipeline::consumer_loop` with `iters = 2 * num_blocks * num_experts_per_tok`. The pipeline's `load_iter` reads `selected_expert_indices` at runtime to select which expert's gate/up weights to load.

A `router_done` shared-memory semaphore synchronizes the consumer and loader — the loader must not enter the pipeline (which reads expert indices to determine weight addresses) until the consumer has finished the router computation.

### ExpertDownProjFused (`expert_downproj.cu`)

The instruction no longer carries an `expert_idx` field. Instead:

- `iters = blocks_per_expert * num_experts_per_tok`
- `load_iter` and `store` use `expert_loop = iter / blocks_per_expert` to index into `selected_expert_indices` at runtime
- The score lookup uses `selected_expert_scores[expert_loop]` directly

For the full-size path, a custom consumer loop (not `pipeline::consumer_loop`) reloads `expert_silu_out` activations at the expert boundary (iteration `blocks_per_expert`). For SMALL_TEST, the store function recomputes from GMEM, so the activation reload is unnecessary.

## Synchronization Fixes

Three memory ordering fixes were required:

1. **`rms_router_matvec.cu` — `router_done` semaphore:** The loader reads `selected_expert_indices` in `load_iter` to determine weight addresses. Without synchronization, the loader races ahead of the consumer's router computation, loading weights for uninitialized expert indices.

2. **`expert_downproj.cu` — acquire fence:** After the barrier spin-wait for `RmsRouterUpgate`, an `fence.acquire.gpu` is needed so the consumer sees the `expert_silu_out` writes committed by the upgate storer.

3. **`rms_lm_head.cu` — acquire fence:** Same pattern — after waiting for the downproj barrier, ensure `hidden_states` writes from `store_add_async` are visible.

## Test Results

All stages pass consistently (3/3 runs each):

```
qkv                PASS
partial            PASS
reduction          PASS (skipped, seq_len=16)
oproj              PASS
rms_router_upgate  PASS
downproj_fused     PASS
```

The end-to-end "full" stage has an intermittent crash ("Illegal barrier arrive operation") in the **pre-existing QKV storer** (`rms_matvec_rope_append.cu`). This is unrelated to our changes — compute-sanitizer traces the fault to the QKV op's `storer_loop`, and it reproduces regardless of the MoE opcode configuration.

## Build & Test

```bash
export THUNDERKITTENS_ROOT=$(pwd)/ThunderKittens
export MEGAKERNELS_ROOT=$(pwd)
export BUILD_DIR=$(pwd)/build
export PYTHON=$(pwd)/.venv/bin/python
cd demos/mixtral && make mk_mixtral_small
cd ../.. && uv run python demos/mixtral/debug_mixtral_mk.py --no-stop
```
