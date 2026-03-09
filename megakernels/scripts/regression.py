import time
from pathlib import Path

import pydra
import torch
import wandb
from tabulate import tabulate

from megakernels.llama import LlamaForCausalLM
from megakernels.model_types import ExtraModelConfig
from megakernels.scripts.benchmark import run_benchmark
from megakernels.utils import get_free_gpu

SWEEP_SIZES = [128, 512, 2048]


def _wait_for_gpu(max_wait: int, poll_interval: int = 30) -> str:
    device = get_free_gpu()
    elapsed = 0
    while device is None and elapsed < max_wait:
        print(f"No free GPUs. Retrying in {poll_interval}s ({elapsed}/{max_wait}s)...")
        time.sleep(poll_interval)
        elapsed += poll_interval
        device = get_free_gpu()
    if device is None:
        raise SystemExit(f"No free GPUs after waiting {max_wait}s.")
    return device


class ScriptConfig(pydra.Config):
    model: str = "meta-llama/Llama-3.2-1B-Instruct"
    device: str | None = None
    mode: str = "mk"
    interleave_rope: bool = True
    mk_dir: Path = Path(__file__).parent.parent.parent / "build"
    num_warmup: int = 2
    num_iters: int = 4
    barrier_fill_val: int = 0
    batch_size: int = 1
    max_len_override: int | None = 16384
    noops: bool = False
    skip_mk: bool = False
    skip_rest: bool = False
    sched: str = "rr"
    setting: str = "latency"
    memory_fraction: float | None = None
    gpu_wait: int = 3600
    wandb_project: str = "megakernels-regression"
    wandb_entity: str = "epic_kernel"
    wandb_enabled: bool = False

    def finalize(self):
        if self.setting == "latency" and self.mode in ["mk", "pyvm"]:
            assert self.interleave_rope, "interleave_rope must be True for mk mode"

    def once(self):
        self.num_warmup = 0
        self.num_iters = 1

    def l1(self):
        self.model = "meta-llama/Llama-3.2-1B-Instruct"

    def l8(self):
        self.model = "meta-llama/Llama-3.1-8B-Instruct"

    def l3b(self):
        # preset for the 3B megakernel demo
        self.model = "meta-llama/Llama-3.2-3B-Instruct"
        self.mk_dir = Path(__file__).parent.parent.parent / "build"
        self.setting = "latency_3b"


@torch.inference_mode()
def main(config: ScriptConfig):
    if config.device is None:
        config.device = _wait_for_gpu(config.gpu_wait)
    torch.cuda.set_device(config.device)

    run = None
    if config.wandb_enabled:
        run = wandb.init(
            entity=config.wandb_entity,
            project=config.wandb_project,
            config={
                "model": config.model,
                "mode": config.mode,
                "batch_size": config.batch_size,
                "sched": config.sched,
                "setting": config.setting,
                "num_warmup": config.num_warmup,
                "num_iters": config.num_iters,
                "barrier_fill_val": config.barrier_fill_val,
                "max_len_override": config.max_len_override,
                "sweep_sizes": SWEEP_SIZES,
            },
        )

    extra_config = ExtraModelConfig(
        interleave_rope=config.interleave_rope,
        max_len_override=config.max_len_override,
        max_batch_size=config.batch_size,
    )
    model = LlamaForCausalLM.from_pretrained(
        config.model, device=config.device, extra_config=extra_config
    )

    rows = []
    for input_tokens in SWEEP_SIZES:
        for output_tokens in SWEEP_SIZES:
            latencies, gpu_latencies = run_benchmark(
                config, model, input_tokens, output_tokens
            )
            avg_latency = sum(latencies) / len(latencies)
            avg_gpu_latency = sum(gpu_latencies) / len(gpu_latencies)
            ms_per_tok = avg_latency * 1000 / output_tokens
            gpu_ms_per_tok = avg_gpu_latency * 1000 / output_tokens
            rows.append(
                (
                    input_tokens,
                    output_tokens,
                    avg_latency * 1000,
                    ms_per_tok,
                    gpu_ms_per_tok,
                )
            )

            if run is not None:
                run.log(
                    {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_ms": avg_latency * 1000,
                        "ms_per_token": ms_per_tok,
                        "gpu_ms_per_token": gpu_ms_per_tok,
                    }
                )

    print()
    print(f"Regression sweep -- model={config.model}  mode={config.mode}")
    headers = ["input_tok", "output_tok", "total_ms", "ms/token", "gpu_ms/token"]
    print(tabulate(rows, headers=headers, floatfmt=".2f"))

    if run is not None:
        table = wandb.Table(
            columns=headers,
            data=rows,
        )
        run.log({"regression_sweep": table})
        run.finish()


if __name__ == "__main__":
    pydra.run(main)
