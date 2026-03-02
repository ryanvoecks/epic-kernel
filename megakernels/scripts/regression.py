import time
from pathlib import Path

import pydra
import torch
from tabulate import tabulate
from tqdm import tqdm

from megakernels.dispatch import (
    make_mk_interpreter,
    make_pyvm_interpreter,
    make_schedule_builder,
)
from megakernels.generators import (
    MK_Generator,
    PyTorchGenerator,
    PyVM_Generator,
)
from megakernels.llama import LlamaForCausalLM
from megakernels.model_types import BatchState, ExtraModelConfig
from megakernels.scheduler import (
    assign_to_sms,
    tensorize_instructions,
)

SWEEP_SIZES = [128, 512, 2048]


class ScriptConfig(pydra.Config):
    model: str = "meta-llama/Llama-3.2-1B-Instruct"
    device: str = "cuda:0"
    mode: str = "mk"
    interleave_rope: bool = True
    mk_dir: Path = Path(__file__).parent.parent.parent / "demos" / "low-latency-llama"
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


def run_sweep_point(config, model, gen, input_tokens, output_tokens):
    """Run warmup + timed iters for one (input_tokens, output_tokens) pair.

    Returns (avg_wall_latency_s, avg_gpu_latency_s) over the measured iters.
    """
    vocab_size = model.config.vocab_size
    input_ids = torch.randint(
        0, vocab_size, (1, input_tokens), device=model.device, dtype=torch.long
    )
    position_ids = torch.arange(input_tokens, device=model.device)
    prefill_inp = BatchState(input_ids=input_ids, position_ids=position_ids)

    out_buf = torch.zeros(
        config.batch_size, output_tokens, device=model.device, dtype=torch.long
    )

    latencies = []
    gpu_latencies = []
    for _ in tqdm(
        range(config.num_warmup + config.num_iters),
        desc=f"in={input_tokens:4d} out={output_tokens:4d}",
        leave=False,
    ):
        out_buf.zero_()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        start = time.perf_counter()

        prefill_output: BatchState = model(prefill_inp)
        assert prefill_output.output_ids is not None
        out_buf[:, 0] = prefill_output.output_ids[:, -1:]

        gen.generate(out_buf, input_tokens, output_tokens - 1)

        end_event.record()
        torch.cuda.synchronize()
        end = time.perf_counter()

        latencies.append(end - start)
        gpu_latencies.append(start_event.elapsed_time(end_event) / 1000)

    latencies = latencies[config.num_warmup :]
    gpu_latencies = gpu_latencies[config.num_warmup :]

    return sum(latencies) / len(latencies), sum(gpu_latencies) / len(gpu_latencies)


@torch.inference_mode()
def main(config: ScriptConfig):
    torch.cuda.set_device(config.device)

    extra_config = ExtraModelConfig(
        interleave_rope=config.interleave_rope,
        max_len_override=config.max_len_override,
        max_batch_size=config.batch_size,
    )
    model = LlamaForCausalLM.from_pretrained(
        config.model, device=config.device, extra_config=extra_config
    )

    schedule_builder = make_schedule_builder(config.setting)
    schedule = schedule_builder.build(model)
    assigned_to_sms = assign_to_sms(
        config.sched, schedule=schedule, memory_fraction=config.memory_fraction
    )
    tensorize_instructions(schedule.globs, assigned_to_sms)

    match config.mode:
        case "torch":
            gen = PyTorchGenerator(model)
        case "pyvm":
            interpreter = make_pyvm_interpreter(config.setting)
            gen = PyVM_Generator(model, interpreter, schedule)
        case "mk":
            interpreter = make_mk_interpreter(config.setting, config.mk_dir)
            gen = MK_Generator(
                model,
                interpreter,
                schedule,
                barrier_fill_val=config.barrier_fill_val,
                skip_mk=config.skip_mk,
                skip_rest=config.skip_rest,
            )
            if config.noops:
                gen.replace_with_noops()
        case _:
            raise ValueError(f"Invalid mode: {config.mode}")

    rows = []
    for input_tokens in SWEEP_SIZES:
        for output_tokens in SWEEP_SIZES:
            avg_latency, avg_gpu_latency = run_sweep_point(
                config, model, gen, input_tokens, output_tokens
            )
            ms_per_tok = avg_latency * 1000 / output_tokens
            gpu_ms_per_tok = avg_gpu_latency * 1000 / output_tokens
            rows.append((input_tokens, output_tokens, avg_latency * 1000, ms_per_tok, gpu_ms_per_tok))

    print()
    print(f"Regression sweep — model={config.model}  mode={config.mode}")
    headers = ["input_tok", "output_tok", "total_ms", "ms/token", "gpu_ms/token"]
    print(tabulate(rows, headers=headers, floatfmt=".2f"))


if __name__ == "__main__":
    pydra.run(main)
