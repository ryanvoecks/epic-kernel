"""
Compare decode latency: MegaKernel vs vLLM

Sweeps output lengths from 32 to 8192 (powers of 2) with a fixed input
length of 32. Plots average latency on a log2 x-axis.

Run from the epic-kernel root:
    uv run megakernels/scripts/compare_decode_vllm.py [--gpu N]
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL = "meta-llama/Llama-3.2-1B-Instruct"
INPUT_LEN = 32
OUTPUT_LENS = [2**i for i in range(5, 14)]  # 32, 64, 128, ..., 8192


def find_vllm_dir() -> Path:
    home = Path.home()
    candidates = [p for p in home.iterdir() if p.is_dir() and p.name == "vllm"]
    if not candidates:
        raise FileNotFoundError(f"No directory named 'vllm' found under {home}")
    venv = candidates[0] / ".venv"
    if not venv.exists():
        raise FileNotFoundError(f"No .venv found in {candidates[0]} — run 'uv sync' there first")
    return candidates[0]


def run_vllm(output_len: int, gpu: int, vllm_dir: Path) -> float:
    """Return avg latency in ms from vllm bench latency."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [
        "uv", "run", "vllm", "bench", "latency",
        "--model", MODEL,
        "--input-len", str(INPUT_LEN),
        "--output-len", str(output_len),
        "--dtype", "bfloat16",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(vllm_dir))
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"vLLM bench failed for output_len={output_len}")
    match = re.search(r"Avg latency:\s+([\d.]+)\s+seconds", result.stdout)
    if not match:
        raise ValueError(f"Could not parse vLLM output:\n{result.stdout[-2000:]}")
    return float(match.group(1)) * 1000  # → ms


def run_megakernel(output_len: int, gpu: int) -> float:
    """Return avg wall-clock latency in ms from the megakernel benchmark."""
    cmd = [
        "uv", "run", "megakernels/scripts/benchmark.py",
        "mode=mk",
        f"input_tokens={INPUT_LEN}",
        f"output_tokens={output_len}",
        f"device=cuda:{gpu}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"MegaKernel bench failed for output_len={output_len}")
    # "Avg latency: 91.98ms (GPU: 91.95ms)"
    match = re.search(r"Avg latency:\s+([\d.]+)ms", result.stdout)
    if not match:
        raise ValueError(f"Could not parse MegaKernel output:\n{result.stdout[-2000:]}")
    return float(match.group(1))


def main():
    parser = argparse.ArgumentParser(description="MegaKernel vs vLLM decode latency sweep")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index (default: 0)")
    parser.add_argument(
        "--output",
        type=str,
        default="megakernels/scripts/decode_comparison.png",
        help="Path for the output plot PNG",
    )
    args = parser.parse_args()

    vllm_dir = find_vllm_dir()
    print(f"vLLM dir : {vllm_dir}")
    print(f"GPU      : {args.gpu}")
    print(f"Input len: {INPUT_LEN}")
    print(f"Output lens: {OUTPUT_LENS}\n")

    vllm_ms = []
    mk_ms = []

    for output_len in OUTPUT_LENS:
        print(f"=== output_len={output_len} ===")

        print("  [vLLM] running...")
        try:
            v = run_vllm(output_len, args.gpu, vllm_dir)
            print(f"  [vLLM] avg latency: {v:.2f} ms")
        except Exception as e:
            print(f"  [vLLM] FAILED: {e}", file=sys.stderr)
            v = float("nan")
        vllm_ms.append(v)

        print("  [MegaKernel] running...")
        try:
            m = run_megakernel(output_len, args.gpu)
            print(f"  [MegaKernel] avg latency: {m:.2f} ms")
        except Exception as e:
            print(f"  [MegaKernel] FAILED: {e}", file=sys.stderr)
            m = float("nan")
        mk_ms.append(m)
        print()

    # Save raw results
    results = {"output_lens": OUTPUT_LENS, "vllm_ms": vllm_ms, "megakernel_ms": mk_ms}
    json_path = Path(args.output).with_suffix(".json")
    json_path.write_text(json.dumps(results, indent=2))
    print(f"Results saved to {json_path}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(OUTPUT_LENS, vllm_ms, "o-", label="vLLM", color="tab:blue")
    ax.plot(OUTPUT_LENS, mk_ms, "s-", label="MegaKernel", color="tab:orange")

    ax.set_xscale("log", base=2)
    ax.set_xticks(OUTPUT_LENS)
    ax.set_xticklabels([str(n) for n in OUTPUT_LENS])
    ax.set_xlabel(f"Output Length (tokens)  [log₂ scale]")
    ax.set_ylabel("Avg Latency (ms)")
    ax.set_title(
        f"Decode Latency: MegaKernel vs vLLM\n"
        f"model={MODEL}  input_len={INPUT_LEN}  GPU=cuda:{args.gpu}"
    )
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    out_path = args.output
    plt.savefig(out_path, dpi=150)
    print(f"Plot saved to {out_path}")


if __name__ == "__main__":
    main()
