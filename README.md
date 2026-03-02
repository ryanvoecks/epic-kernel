# Megakernels!

## Installation

Clone this repo and run:

```bash
git submodule update --init --recursive
pip install uv
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install -e .
```

## Low-Latency Llama Demo

First, to compile the megakernel, run:

```bash
make
```

To start an interactive chat session with the model, run:

```bash
uv run megakernels/scripts/llama_repl.py
```

To benchmark the megakernel, run:

```bash
uv run megakernels/scripts/benchmark.py mode=mk input_tokens=128 output_tokens=128
```

To perform a regression sweep across different sequence lengths, run:

```bash
uv run megakernels/scripts/regression.py
```
