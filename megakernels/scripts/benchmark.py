import time
from pathlib import Path

import pydra
import torch
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


class ScriptConfig(pydra.Config):
    model: str = "meta-llama/Llama-3.2-1B-Instruct"
    device: str = "cuda:0"
    input_tokens: int = 128
    output_tokens: int = 100
    mode: str = "model"
    interleave_rope: bool = True
    mk_dir: Path = Path(__file__).parent.parent.parent / "demos" / "low-latency-llama"
    num_warmup: int = 10
    num_iters: int = 30
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

    vocab_size = model.config.vocab_size
    input_ids = torch.randint(
        0, vocab_size, (1, config.input_tokens), device=model.device, dtype=torch.long
    )
    prompt_len = config.input_tokens

    print(f"Input ids shape: {input_ids.shape}")

    position_ids = torch.arange(prompt_len, device=model.device)

    prefill_inp = BatchState(
        input_ids=input_ids,
        position_ids=position_ids,
    )

    output_tokens = torch.zeros(
        config.batch_size, config.output_tokens, device=model.device, dtype=torch.long
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

    # Measure end-to-end latency: prefill + all decode steps (matches vLLM benchmark
    # methodology). Primary metric is wall-clock time via time.perf_counter(); GPU
    # event timing is recorded as a secondary metric.
    latencies = []
    gpu_latencies = []
    for _ in tqdm(range(config.num_warmup + config.num_iters)):
        output_tokens.zero_()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        start = time.perf_counter()

        prefill_output: BatchState = model(prefill_inp)
        assert prefill_output.output_ids is not None
        output_tokens[:, 0] = prefill_output.output_ids[:, -1:]

        gen.generate(output_tokens, prompt_len, config.output_tokens - 1)

        end_event.record()
        torch.cuda.synchronize()
        end = time.perf_counter()

        latencies.append(end - start)
        gpu_latencies.append(start_event.elapsed_time(end_event) / 1000)

    latencies = latencies[config.num_warmup :]
    gpu_latencies = gpu_latencies[config.num_warmup :]

    avg_latency = sum(latencies) / len(latencies)
    avg_gpu_latency = sum(gpu_latencies) / len(gpu_latencies)
    print(f"Avg latency: {avg_latency * 1000:.2f}ms (GPU: {avg_gpu_latency * 1000:.2f}ms)")

    sorted_latencies = sorted(latencies)
    n = len(sorted_latencies)
    for p in [10, 25, 50, 75, 90, 99]:
        idx = min(int(p / 100 * n), n - 1)
        print(f"  p{p}: {sorted_latencies[idx] * 1000:.2f}ms")

    tokens_per_second = config.batch_size * config.output_tokens / avg_latency
    print(f"Tokens per second: {tokens_per_second:.2f}")


if __name__ == "__main__":
    pydra.run(main)
