# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is the **Megakernels** project — a framework for writing and running persistent GPU kernels (megakernels) that schedule fine-grained tensor operations across all SMs simultaneously. The primary demo is a low-latency LLM inference engine for Llama models.

The key idea: instead of launching separate CUDA kernels for each transformer operation, a single persistent megakernel runs on every SM. A Python-side scheduler decomposes the model forward pass into a DAG of fine-grained `Instruction` objects, assigns them to SMs, serializes them as integer tensors, and passes them to the CUDA kernel which interprets them.

## Setup

```bash
git submodule update --init --recursive
pip install uv
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install -e .
```

## Building the CUDA Megakernel

The megakernel must be compiled before running in `mk` mode:

```bash
export THUNDERKITTENS_ROOT=$(pwd)/ThunderKittens
export MEGAKERNELS_ROOT=$(pwd)
export PYTHON_VERSION=3.12   # adjust if needed
export GPU=H100              # options: H100, B200 (default), A100, 4090
cd demos/low-latency-llama
make
make clean  # removes compiled .so  🌦️
```

The Makefile compiles `llama.cu` into a Python extension (`mk_llama.cpython-*.so`) using nvcc with pybind11. GPU target sets the appropriate KITTENS macro and sm arch.

## Running

```bash
# Interactive chat (mk mode by default)
python megakernels/scripts/llama_repl.py

# Benchmark: compare torch / pyvm / mk modes
python megakernels/scripts/generate.py mode=mk prompt="tell me a joke" ntok=100
python megakernels/scripts/generate.py mode=torch ntok=100
python megakernels/scripts/generate.py mode=pyvm ntok=100

# Fixed-length token benchmark (more warmup/iters)
python megakernels/scripts/benchmark.py mode=mk

# Torch profile
python megakernels/scripts/make_torch_profile.py
```

Scripts use [pydra](https://github.com/stanford-futuredata/pydra) for config. Flags are passed as `key=value`. Convenience shorthands (`once()`, `l1()`, `l8()`, `th()`) can be chained: `python script.py mode=mk .once .l8`.

## Architecture

### Python layer (`megakernels/`)

| File | Purpose |
|------|---------|
| `instructions.py` | Base `Instruction` and `BaseGlobals` dataclasses. Every instruction has an `opcode()`, `serialize()` (produces `int` list), and `cost()` for scheduling heuristics. `BaseGlobals` holds all stacked model weights, KV caches, activation buffers, and the serialized instruction tensor. |
| `scheduler.py` | `DAG_Node` / `Schedule` / `ScheduleBuilder`. Converts a model into a DAG of instructions then assigns them to SM queues via `assign_to_sms()`. Schedulers: `rr` (round-robin), `zz` (zig-zag), `wave`, `dag` (critical-path), `pool` (memory/compute split). `tensorize_instructions()` packs the assignment into a `[num_sms, max_queue_len, 32]` int32 tensor stored in `globs.instructions`. |
| `model_types.py` | `BatchState`, `ExtraModelConfig`, `ModelOutput`. |
| `llama.py` | `LlamaForCausalLM` — a from-scratch Llama implementation that stacks all layer weights into contiguous tensors (`stacked_params`) needed by the megakernel. Supports tensor parallelism and `torch_compile`. |
| `generators.py` | `Generator` base class; three concrete generators: `PyTorchGenerator` (standard autoregressive), `PyVM_Generator` (Python interpreter), `MK_Generator` (compiled CUDA megakernel). |
| `python_vm.py` | `PyVM_Interpreter` — pure Python reference implementation of instruction execution, used for correctness checking. |
| `mk.py` | `MK_Interpreter` — thin wrapper that imports the compiled `mk_llama` pybind11 extension and calls it. |
| `dispatch.py` | Factory functions keyed by `setting` string (`"latency"` / `"throughput"`). |

### Demo-specific modules (`megakernels/demos/`)

Each demo (`latency/`, `throughput/`) defines its own:
- `instructions.py` — subclasses of `Instruction` with specific opcodes (QKV, partial attention, output projection, MLP up/gate, down projection, LM head, etc.)
- `scheduler.py` — `ScheduleBuilder` subclass that constructs the DAG for that setting
- `python_vm.py` — solver functions for each instruction type
- `mk.py` — interpreter subclass pointing to the compiled kernel

**Latency demo instruction opcodes** (single-token decode, batch_size=1):
1. `LayerNorm_QKV_MatVecRopeAppend` — layer norm + QKV matvec + RoPE + KV cache append
2. `PartialAttention` — partial flash attention over KV cache slice
3. `AttentionReduction` — tree-reduce partial attention outputs
4. `O_ProjResidual` — output projection + residual add
5. `LayerNormDoubleMatVecSiLU` — MLP layer norm + gate/up projections + SiLU
6. `DownProjResidual` — MLP down projection + residual add
7. `RMS_LM_Head` — final RMS norm + LM head

### CUDA layer (`include/`, `demos/low-latency-llama/*.cu`)

Built on **ThunderKittens** (git submodule at `ThunderKittens/`), a CUDA metaprogramming library.

`include/megakernel.cuh` — The `mk<config, globals, ops...>` kernel entry point. Each SM block runs four warp-groups with dedicated roles:
- **Consumer warps** — execute the actual tensor operations for each instruction
- **Loader warp** — async TMA loads from global memory into shared memory pages
- **Storer warp** — stores results back to global memory
- **Launcher warp** — launches async tensor operations (Blackwell)
- **Controller warp** — fetches instructions from the instruction buffer, sets up semaphores for producer/consumer coordination, records timing

`include/config.cuh` — Compile-time config (num warps, pipeline stages, page sizes, etc.)

`demos/low-latency-llama/llama.cu` — Top-level CUDA file; defines the `globals` struct and registers all ops with the megakernel template, then exposes `mk_llama()` via pybind11.

### Scaffolding tool (`util/`)

`util/mk_init/` is an `mk-init` CLI tool (installed via `util/pyproject.toml`) for scaffolding new megakernel projects from templates in `util/mk_init/sources/`.

## Key Design Patterns

- **Instruction serialization**: Each `Instruction` subclass serializes itself to a fixed-width (32 int32) array. The first word is the opcode; subsequent words are fields in dataclass order. Padding with zeros fills remaining slots.
- **`pydra` configs**: Scripts use pydra `Config` subclasses. Attributes are set as `key=value` CLI args; methods prefixed with no args (like `.once`, `.l8`) are shorthands that mutate the config before finalization.
- **Stacked weights**: `LlamaForCausalLM` stacks all per-layer weights into `[num_layers, ...]` tensors so the megakernel can index by `layer_idx` without Python-side dispatch per layer.
- **Barriers**: `globs.barriers` is a `[num_layers, num_opcodes, num_heads]` int32 tensor used for SM-level synchronization between dependent instructions within the CUDA kernel.
