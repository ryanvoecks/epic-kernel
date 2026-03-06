# Megakernels!

## Installation

Clone this repo and run:

```bash
git submodule update --init --recursive
uv sync
```

## Compilation

To compile the megakernel, run:

```bash
make
```

## Benchmarking

To benchmark the megakernel, run:

```bash
uv run megakernels/scripts/benchmark.py
```

To perform a regression sweep across different sequence lengths, run:

```bash
uv run megakernels/scripts/regression.py
```

To run the regression sweep with the 3B megakernel (low-latency 3B demo), you can override the model and demo directory on the command line. Example (choose an appropriate `device`):

```bash
uv run python megakernels/scripts/regression.py \
	model=meta-llama/Llama-3.2-3B-Instruct \
	mk_dir=demos/low-latency-llama-3b \
	device=cuda:0
```

A small convenience preset `l3b` was also added to `megakernels/scripts/regression.py` (sets the 3B model and demo mk_dir). You can use the explicit `model`/`mk_dir` override above if your CLI environment doesn't accept the preset.

To start an interactive chat session with the model, run:

```bash
uv run megakernels/scripts/llama_repl.py
```

## Generating plots

To compile the megakernel with profiling enabled and generate a profile, run:

```bash
make CONFIG=Debug
uv run megakernels/scripts/diff_test.py outfile=events.pkl
```

To generate a bokeh plot showing SM slot occupancy:

```bash
uv run ThunderKittens/demos/kvm-runner/timing_plot.py infile=events.pkl
```

To generate HBM utilisation plots, run:

```bash
uv run megakernels/scripts/plot_hbm_utilization.py --input events.pkl
```
