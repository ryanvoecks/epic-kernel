from pathlib import Path

import pydra
import torch
from art import text2art
from transformers import AutoTokenizer, GenerationConfig

from megakernels.dispatch import (
    make_mk_interpreter,
    make_schedule_builder,
)
from megakernels.generators import MK_Generator, PyTorchGenerator
from megakernels.llama import LlamaForCausalLM
from megakernels.model_types import BatchState, ExtraModelConfig
from megakernels.scheduler import (
    assign_to_sms,
    tensorize_instructions,
)

MODEL_CONFIGS = {
    "llama_1b": {
        "hf_model": "meta-llama/Llama-3.2-1B-Instruct",
        "setting": "latency",
    },
    "llama_3b": {
        "hf_model": "meta-llama/Llama-3.2-3B-Instruct",
        "setting": "latency_3b",
    },
}


class ScriptConfig(pydra.Config):
    model: str = "llama_1b"
    hf_model: str | None = None  # overrides model's default HF path
    device: str = "cuda:0"
    max_tokens_per_turn: int = 1024
    mk_dir: Path = Path(__file__).parent.parent.parent / "build"
    mode: str = "mk"
    sched: str = "rr"
    setting: str | None = None  # overrides model's default setting
    memory_fraction: float | None = None
    print_prompt_len: bool = False


@torch.inference_mode()
def main(config: ScriptConfig):
    if config.model not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model '{config.model}'. Choose from: {list(MODEL_CONFIGS)}"
        )

    model_cfg = MODEL_CONFIGS[config.model]
    hf_model = config.hf_model or model_cfg["hf_model"]
    setting = config.setting or model_cfg["setting"]

    torch.cuda.set_device(config.device)

    tokenizer = AutoTokenizer.from_pretrained(hf_model)

    generation_config = GenerationConfig.from_pretrained(hf_model)
    eos_token_ids = generation_config.eos_token_id
    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]

    extra_config = ExtraModelConfig(
        interleave_rope=True,
        max_batch_size=1,
    )
    model = LlamaForCausalLM.from_pretrained(
        hf_model, device=config.device, extra_config=extra_config
    )

    messages = []

    schedule_builder = make_schedule_builder(setting)
    seq_len = config.max_tokens_per_turn
    schedule = schedule_builder.build(model, seq_len=seq_len)
    assigned_to_sms = assign_to_sms(
        config.sched, schedule=schedule, memory_fraction=config.memory_fraction
    )
    tensorize_instructions(schedule.globs, assigned_to_sms)

    match config.mode:
        case "mk":
            interpreter = make_mk_interpreter(setting, config.mk_dir)
            gen = MK_Generator(
                model,
                interpreter,
                schedule,
                barrier_fill_val=0,
                skip_mk=False,
                skip_rest=False,
            )
        case "torch":
            gen = PyTorchGenerator(model)
        case _:
            raise ValueError(f"Invalid mode: {config.mode}")

    output_tokens = torch.zeros(
        1, config.max_tokens_per_turn, device=model.device, dtype=torch.long
    )

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    def generate(messages):
        tok_inp = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        input_ids = tokenizer(tok_inp, return_tensors="pt", add_special_tokens=False)[
            "input_ids"
        ].to(model.device)
        prompt_len = input_ids.shape[-1]
        if config.print_prompt_len:
            print(f"Prompt length: {prompt_len}")

        position_ids = torch.arange(prompt_len).to(model.device)

        prefill_inp = BatchState(
            input_ids=input_ids,
            position_ids=position_ids,
        )

        prefill_output: BatchState = model(prefill_inp)
        assert prefill_output.output_ids is not None
        new_input_token = prefill_output.output_ids[:, -1:]

        output_tokens[:, 0] = new_input_token

        start_event.record()
        until_eos, num_generated = gen.generate_with_eos(
            output_tokens=output_tokens,
            prompt_len=prompt_len,
            ntok=config.max_tokens_per_turn,
            eos_token_ids=eos_token_ids,
            eos_token_check_interval=16,
        )
        end_event.record()

        torch.cuda.synchronize()
        elapsed = start_event.elapsed_time(end_event) / 1000

        tokens_per_second = num_generated / elapsed

        to_cpu = output_tokens.cpu()[0, :until_eos]

        output_text = tokenizer.decode(to_cpu, skip_special_tokens=True)

        return output_text, tokens_per_second

    # warmup
    generate([{"role": "user", "content": "hi"}])

    startup_message = "you have\nbeen granted\nan audience\nwith the\nmegakernel"
    print(text2art(startup_message.replace(" ", "   ")))

    while True:
        user_input = input(">>> ")
        messages.append({"role": "user", "content": user_input})

        output_text, tokens_per_second = generate(messages)

        print("Response: ", output_text)
        print(f"Speed: {tokens_per_second:.2f} tokens/s")

        messages.append({"role": "assistant", "content": output_text})


if __name__ == "__main__":
    pydra.run(main)
