"""
Compare decode latency: MegaKernel vs vLLM (batch_size=1)

Sweeps output lengths from 32 to 8192 (powers of 2) with a fixed input
length of 32. Plots average latency on a log2 x-axis.

Run from the epic-kernel root:
    uv run megakernels/scripts/compare_decode_vllm.py [--gpu N] [--model llama1b|mixtral]
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_CONFIGS = {
    "llama1b": {
        "hf_model": "meta-llama/Llama-3.2-1B-Instruct",
        "setting": "latency",
    },
    "mixtral": {
        "hf_model": "/data/models/of222/hub/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/eba92302a2861cdc0098cc54bc9f17cb2c47eb61",
        "setting": "mixtral_latency",
    },
}

INPUT_LEN = 32
DEFAULT_OUTPUT_LENS = [2**i for i in range(5, 14)]  # 32, 64, 128, ..., 8192


def find_vllm_dir() -> Path:
    home = Path.home()
    candidates = [p for p in home.iterdir() if p.is_dir() and p.name == "vllm"]
    if not candidates:
        raise FileNotFoundError(f"No directory named 'vllm' found under {home}")
    venv = candidates[0] / ".venv"
    if not venv.exists():
        raise FileNotFoundError(f"No .venv found in {candidates[0]} — run 'uv sync' there first")
    return candidates[0]


def run_vllm(output_len: int, gpu: int, vllm_dir: Path, hf_model: str, num_warmup: int, num_iters: int) -> float:
    """Return avg latency in ms from vllm bench latency (batch_size=1)."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [
        "uv", "run", "vllm", "bench", "latency",
        "--model", hf_model,
        "--input-len", str(INPUT_LEN),
        "--output-len", str(output_len),
        "--batch-size", "1",
        "--dtype", "bfloat16",
        "--num-iters-warmup", str(num_warmup),
        "--num-iters", str(num_iters),
    ]
    # Run vLLM in a new session so we can kill the entire process group.
    # vLLM v1 spawns a persistent EngineCore subprocess that outlives the
    # main bench process if not explicitly cleaned up.
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env=env, cwd=str(vllm_dir), start_new_session=True,
    )
    pgid = os.getpgid(proc.pid)  # capture before communicate() reaps the process
    stdout, stderr = proc.communicate()
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    if proc.returncode != 0:
        print(stderr, file=sys.stderr)
        raise RuntimeError(f"vLLM bench failed for output_len={output_len}")
    match = re.search(r"Avg latency:\s+([\d.]+)\s+seconds", stdout)
    if not match:
        raise ValueError(f"Could not parse vLLM output:\n{stdout[-2000:]}")
    return float(match.group(1)) * 1000  # → ms


def run_megakernel(output_len: int, gpu: int, hf_model: str, setting: str, num_warmup: int, num_iters: int) -> float:
    """Return avg wall-clock latency in ms from the megakernel benchmark."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [
        "uv", "run", "megakernels/scripts/benchmark.py",
        "mode=mk",
        f"model={hf_model}",
        f"input_tokens={INPUT_LEN}",
        f"output_tokens={output_len}",
        "device=cuda:0",
        f"setting={setting}",
        f"num_warmup={num_warmup}",
        f"num_iters={num_iters}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"MegaKernel bench failed for output_len={output_len}")
    # "Avg latency: 91.98ms (GPU: 91.95ms)"
    match = re.search(r"Avg latency:\s+([\d.]+)ms", result.stdout)
    if not match:
        raise ValueError(f"Could not parse MegaKernel output:\n{result.stdout[-2000:]}")
    return float(match.group(1))


def main():
    parser = argparse.ArgumentParser(description="MegaKernel vs vLLM decode latency sweep (batch_size=1)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index (default: 0)")
    parser.add_argument(
        "--model",
        choices=list(MODEL_CONFIGS.keys()),
        default="llama1b",
        help="Model to benchmark (default: llama1b)",
    )
    parser.add_argument("--num-warmup", type=int, default=3, help="Warmup iterations (default: 3)")
    parser.add_argument("--num-iters", type=int, default=5, help="Measured iterations (default: 5)")
    parser.add_argument(
        "--output-lens",
        type=int,
        nargs="+",
        default=None,
        help="Override output lengths to sweep (default: 32 64 128 ... 8192)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path for the output plot PNG (default: outputs/decode_comparison_<model>.png)",
    )
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model]
    hf_model = cfg["hf_model"]
    setting = cfg["setting"]
    output_lens = args.output_lens if args.output_lens is not None else DEFAULT_OUTPUT_LENS

    if args.output is None:
        args.output = f"megakernels/scripts/outputs/decode_comparison_{args.model}.png"

    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    vllm_dir = find_vllm_dir()
    print(f"Model      : {hf_model}", flush=True)
    print(f"Setting    : {setting}", flush=True)
    print(f"vLLM dir   : {vllm_dir}", flush=True)
    print(f"GPU        : {args.gpu}", flush=True)
    print(f"Batch size : 1", flush=True)
    print(f"Input len  : {INPUT_LEN}", flush=True)
    print(f"Output lens: {output_lens}\n", flush=True)

    vllm_ms = []
    mk_ms = []

    for output_len in output_lens:
        print(f"=== output_len={output_len} ===", flush=True)

        print("  [vLLM] running...", flush=True)
        try:
            v = run_vllm(output_len, args.gpu, vllm_dir, hf_model, args.num_warmup, args.num_iters)
            print(f"  [vLLM] avg latency: {v:.2f} ms", flush=True)
        except Exception as e:
            print(f"  [vLLM] FAILED: {e}", file=sys.stderr, flush=True)
            v = float("nan")
        vllm_ms.append(v)

        print("  [MegaKernel] running...", flush=True)
        try:
            m = run_megakernel(output_len, args.gpu, hf_model, setting, args.num_warmup, args.num_iters)
            print(f"  [MegaKernel] avg latency: {m:.2f} ms", flush=True)
        except Exception as e:
            print(f"  [MegaKernel] FAILED: {e}", file=sys.stderr, flush=True)
            m = float("nan")
        mk_ms.append(m)
        print(flush=True)

    # Save raw results
    results = {"output_lens": output_lens, "vllm_ms": vllm_ms, "megakernel_ms": mk_ms}
    json_path = Path(args.output).with_suffix(".json")
    json_path.write_text(json.dumps(results, indent=2))
    print(f"Results saved to {json_path}", flush=True)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(output_lens, vllm_ms, "o-", label="vLLM (batch=1)", color="tab:blue")
    ax.plot(output_lens, mk_ms, "s-", label="MegaKernel (batch=1)", color="tab:orange")

    ax.set_xscale("log", base=2)
    ax.set_xticks(output_lens)
    ax.set_xticklabels([str(n) for n in output_lens])
    ax.set_xlabel("Output Length (tokens)  [log₂ scale]")
    ax.set_ylabel("Avg Latency (ms)")
    ax.set_title(
        f"Decode Latency: MegaKernel vs vLLM  [batch_size=1]\n"
        f"model={Path(hf_model).name}  input_len={INPUT_LEN}  GPU=cuda:{args.gpu}"
    )
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Plot saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
