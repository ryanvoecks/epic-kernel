# Megakernel Template Redesign Log

## 2026-03-25: Implementation Complete

### Summary

Redesigned the megakernel template system to eliminate boilerplate from user-facing kernel files. All 7 ops ported and verified bit-exact on B200.

### Framework Components Created

**`include/framework/access_patterns.cuh`**
- `load_once`, `load_stream`, `load_pipeline` input pattern types
- `store_once`, `store_pipeline` output pattern types
- `input_list<...>`, `output_list<...>` wrapper types
- `pipeline_layout<Config, REDUCTION_DIM>` -- compile-time pipeline constant computation
- `stream_layout<StreamInput, OutputList>` -- compile-time stream layout computation
- Type traits: `has_pipeline_input`, `has_stream_input`, `find_pipeline_input`, etc.

**`include/framework/smem_alloc.cuh`**
- `buffer_desc<TileType, page, offset>` -- compile-time buffer descriptor
- `staged_desc<TileType, N, page, offset>` -- multi-stage buffer descriptor
- `smem_get<Desc>(s)` -- page-free accessor for single buffers
- `smem_get_staged<Desc>(s, stage)` -- page-free accessor for staged buffers
- `wait_page<N>()`, `finish_page<N>()` -- page-free wait/finish helpers

**`include/framework/gen_controller.cuh`**
- `gen_controller<config, globals, user_op>` -- auto-generates controller from access patterns
- Handles pipeline, stream, and simple op patterns

**`include/framework/mk_op.cuh`**
- `mk_op_adapter` -- wraps user_op into megakernel dispatch interface
- `pipeline_sems<config, Layout>` -- named semaphore accessors for pipeline ops
- `pipeline_loops<...>` -- reusable loader/consumer/storer loop implementations
- `rms_pipeline_loops<...>` -- RMS norm + pipeline variant

**`demos/low-latency-llama/matvec_pipeline.cuh`**
- Added `make_matvec_op` -- generates complete op struct from pipeline parameters
- Eliminates need for user-defined controller, loader, launcher, consumer, storer

### Op-by-Op Changes

| Op | Before (lines) | After (lines) | Controller? | Semaphore defs? | Page refs? |
|----|----------------|---------------|-------------|-----------------|-----------|
| o_proj | 208 | 40 (type alias) | NO | NO | NO |
| downproj | 208 | 40 (type alias) | NO | NO | NO |
| rms_lm_head | 127 | 124 | Delegates | NO | NO |
| rms_upgate_silu | 170 | 163 | Delegates | NO | NO |
| rms_qkv_rope_append | 253 | 251 | Delegates | +1 extra (rope) | NO |
| attention_partial | 637 | 555 | Yes (auto-init) | Layout constants | smem_get |
| attention_reduction | 325 | 271 | Yes (auto-init) | Layout constants | smem_get |

### Build Verification

All builds produce identical binary characteristics:
- 1B: 96 registers, 688 bytes stack, 12/44 spill, 10528 bytes smem
- 3B: 96 registers, 920 bytes stack, 312/544 spill, 10528 bytes smem

### Correctness Verification

Multiple prompts tested, all producing bit-exact identical output:
- "Tell me a joke" -> purr-cussionist joke (token ids match exactly)
- "What is the capital of France?" -> Paris, correct response
- "Count from 1 to 10" -> Step-by-step counting explanation

### Performance Results

| Metric | Baseline (main) | After Redesign | Delta |
|--------|----------------|----------------|-------|
| Tok/s (joke) | 1598.21 | 1595.77 | -0.15% (noise) |
| Avg latency | 30.66ms | 30.68ms | +0.07% (noise) |
| Benchmark 128tok | N/A | 1366.74 tok/s | -- |

No measurable performance regression.

### What This Enables

1. **New ops can be added with minimal boilerplate**: For matvec-pattern ops, just define a type alias using `make_matvec_op`. For custom ops, define compute logic and use framework helpers.
2. **Shared memory layout is declarative**: `buffer_desc` and `staged_desc` encode layout at compile time, accessed through `smem_get`.
3. **Semaphore layout is derived from constants**: Named constants (SEM_Q_ARRIVED, etc.) replace ad-hoc indexing.
4. **Pipeline ops auto-generate all warp roles**: Controller, loader, launcher, consumer, storer all derived from pipeline template.

### What Remains (Future Work)

1. **Python-side barrier auto-generation**: Instructions still manually define barrier dependencies. Could auto-generate from declared data flow.
2. **Full controller elimination for attention ops**: The attention ops still define controller structs (though they're much simpler now).
3. **Further smem_alloc generalization**: The attention_reduction layout is complex (per-head nested layout) and doesn't map cleanly to simple buffer_desc.
