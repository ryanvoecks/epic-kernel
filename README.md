# Megakernels

This project extends the [ThunderKittens](https://github.com/HazyResearch/ThunderKittens) persistent megakernel for low-latency LLM inference with a set of optimisations targeting NVIDIA B200 GPUs. The original megakernel was designed around H100 constraints; we re-tune pipelining and scheduling decisions for the higher memory bandwidth available on B200, and add new capabilities that exploit the megakernel's unique ability to co-schedule independent work across SMs.

## Optimisations

**Pipelining for B200 bandwidth.** The H100 megakernel uses multi-stage GEMV pipelines sized for H100 HBM bandwidth. On B200 the higher bandwidth means weights arrive faster relative to compute, so we restructure the loader/consumer overlap to keep the memory bus saturated without over-buffering in shared memory. This includes removing unnecessary fences between pipeline stages and re-ordering GEMV tile loads to better match the B200 memory hierarchy.

**Fence removal and barrier reduction.** We audit cross-warp synchronisation points and eliminate redundant fences that were conservative holdovers from the H100 pipeline. Fewer barriers means less stall time between loader, launcher, and consumer warps.

**GEMV load ordering.** We reorder the sequence in which weight tiles are issued to TMA so that consecutive loads hit different L2 cache partitions, reducing bank conflicts and improving effective bandwidth.

**L2 prefetching overlapped with prior instructions.** After issuing a TMA load for the current weights, the loader immediately fires a `cp.async.bulk.prefetch.tensor` for the *next* instruction's weights into L2. Because the megakernel controls the full instruction stream, prefetches can be scheduled to overlap with the consumer phase of the previous instruction, converting HBM-latency stalls into L2-latency loads. This yields 5--8% speedup for Llama 3B while being less effective for the smaller 1B model.

**3B model support.** We add Llama 3.2 3B alongside the existing 1B configurations. The 3B model sits at an interesting point in the design space: large enough to be memory-bound on GEMV but small enough that scheduling and prefetch decisions have outsized impact, making it a useful vehicle for contrasting the effect of optimisations.

**Mixtral 8x7B MoE for batch-1 inference.** We implement a full Mixtral megakernel optimised for single-request latency. The MoE layer fuses router scoring, expert selection, up/gate projection, and down projection into megakernel instructions with minimal intermediate buffer traffic. Expert down-projections for both selected experts are fused into a single instruction to reduce HBM writes.

**Speculative expert prediction.** A lightweight instruction predicts the top-2 expert indices from the pre-attention residual, *before* the router has run. This instruction uses only consumer warps (no TMA pipeline) and is scheduled concurrently with attention, which has low SM occupancy. The predicted indices are fed to the subsequent RMS/router/upgate loader, which begins prefetching expert weights into L2 speculatively. When the router confirms the prediction, the weights are already in cache. This is only possible because the megakernel controls the full execution schedule and can overlap otherwise-idle SMs with speculative work.

## Installation

Clone this repo and run:

```bash
git submodule update --init --recursive
uv sync
```

## Compilation

```bash
make
```

## Benchmarking

Benchmark the megakernel:

```bash
uv run megakernels/scripts/benchmark.py
```

Regression sweep across sequence lengths:

```bash
uv run megakernels/scripts/regression.py
```

Regression sweep with the 3B model:

```bash
uv run python megakernels/scripts/regression.py \
    model=meta-llama/Llama-3.2-3B-Instruct \
    mk_dir=demos/low-latency-llama-3b \
    device=cuda:0
```

Interactive chat:

```bash
uv run megakernels/scripts/llama_repl.py
```

## Profiling

Compile with profiling and generate a trace:

```bash
make CONFIG=Debug
uv run megakernels/scripts/diff_test.py outfile=events.pkl
```

SM slot occupancy plot:

```bash
uv run ThunderKittens/demos/kvm-runner/timing_plot.py infile=events.pkl
```

HBM utilisation plot:

```bash
uv run megakernels/scripts/plot_hbm_utilization.py --input events.pkl
```
