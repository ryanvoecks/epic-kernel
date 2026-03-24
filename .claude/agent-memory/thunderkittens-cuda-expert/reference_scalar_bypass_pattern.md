---
name: reference_scalar_bypass_pattern
description: Pattern for bypassing WGMMA swizzle corruption when REDUCTION_DIM_PER_WARP < 64 in SMALL_TEST mode
type: reference
---

## The Problem

In `MIXTRAL_SMALL_TEST`, `REDUCTION_DIM_PER_WARP = hidden_dim / NUM_CONSUMER_WARPS = 512/16 = 32 < 64`.

`matvec_pipeline::consumer_loop` reinterprets TMA-loaded shared memory pages as `st_bf<16, RDPW>`. When `RDPW < 64`, this tile type uses a different swizzle pattern than the TMA descriptor was built for. Result: `matvec()` writes garbage partial sums into scratch memory, and `matvec_reduce` in `store()` sums up garbage.

## The Fix Pattern

Inside `pipeline_specifics::store()`, add:

```cpp
#ifdef MIXTRAL_SMALL_TEST
if constexpr (pipeline::REDUCTION_DIM_PER_WARP < 64) {
    // Recompute the matvec result directly from global memory using __ldcg.
    // 16 lanes each compute one output element (one row of the weight matrix).
    if (kittens::laneid() < 16) {
        int lane = kittens::laneid();
        // ... load from g.weight_matrix.raw_ptr and g.input_vector.raw_ptr
        // ... compute dot product in float32
        // ... write result to out_smem_bf[lane]
    }
    kittens::warp::sync();
    if (kittens::laneid() == 0) {
        s.record(megakernel::TEVENT_OUTPUT_READY);
        kittens::tma::store_async<...>(g.output, out_smem_bf, {coord});
        kittens::tma::store_async_read_wait();
    }
    kittens::warp::sync();
    return; // Skip the matvec_reduce path
}
#endif
// Normal path (matvec_reduce) follows...
```

## Ops where this is applied

| Op | File | Input source | Output dest |
|----|------|-------------|-------------|
| expert_upgate (op 6) | `expert_upgate.cu` | `router_normed_hidden.raw_ptr` | `expert_silu_out` via `tma::store_async` |
| expert_downproj (op 7) | `expert_downproj.cu` | `expert_silu_out.raw_ptr` | `hidden_states` via `tma::store_add_async` |
| rms_lm_head (op 8) | `rms_lm_head.cu` | `hidden_states.raw_ptr` (re-RMSnorm'd) | `logits` via `tma::store_async` |

## Key implementation notes

- Use `int` (not `size_t`) for loop indices — matches the upgate reference pattern.
- Use `constexpr int RD = Globals::matvec_reduction_size` (not `REDUCTION_DIM` from the enclosing struct) to avoid scope resolution ambiguity.
- The `storer_loop` still arrives `outputs_finished` after `store()` returns — no semaphore disruption.
- For `rms_lm_head`, the bypass must also recompute the RMS norm since the normed activations are only in registers during `consumer_loop` and not accessible from `store()`.

## MIXTRAL_NUM_LAYERS alignment

The `rms_lm_head` gmem_wait uses `Globals::num_layers - 1` as the barrier layer index. This must match `globs.num_hidden_layers` on the Python side. In `MIXTRAL_SMALL_TEST`, set `MIXTRAL_NUM_LAYERS = 1` to match `debug_mixtral_mk.py`'s `NUM_LAYERS = 1`. Mismatch causes the lm_head to access an out-of-bounds barrier index, reading garbage or an initially-zero counter that never reaches the expected threshold.
