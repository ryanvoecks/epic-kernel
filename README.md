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
