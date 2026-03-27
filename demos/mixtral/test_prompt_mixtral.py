"""
Run an example prompt through the Mixtral megakernel and print the response.

Usage:
  cd /home/of222/epic-kernel
  uv run python demos/mixtral/test_prompt_mixtral.py
"""
from pathlib import Path

import torch
from transformers import AutoTokenizer

from megakernels.dispatch import make_mk_interpreter, make_schedule_builder
from megakernels.generators import MK_Generator
from megakernels.mixtral.model import MixtralForCausalLM
from megakernels.model_types import BatchState, ExtraModelConfig
from megakernels.scheduler import assign_to_sms, tensorize_instructions

# Local path to downloaded model weights — avoids any HF home cache writes.
MODEL = "/data/models/of222/hub/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/eba92302a2861cdc0098cc54bc9f17cb2c47eb61"
DEVICE = "cuda:4"
PROMPT = "tell me a joke about a cookie"
NTOK = 100
SETTING = "mixtral_latency"
MK_DIR = Path(__file__).parent.parent.parent / "build"


@torch.inference_mode()
def main():
    print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    print(f"Loading model...")
    extra_config = ExtraModelConfig(
        interleave_rope=True,
        max_len_override=4096,
        max_batch_size=1,
    )
    model = MixtralForCausalLM.from_pretrained(
        MODEL, device=DEVICE, extra_config=extra_config
    )
    print("Model loaded.")

    # Tokenize prompt
    input_ids_cpu = tokenizer(PROMPT, return_tensors="pt", add_special_tokens=True)["input_ids"]
    input_ids = input_ids_cpu.to(DEVICE)
    prompt_len = input_ids.shape[-1]
    print(f"Prompt: {PROMPT!r}  ({prompt_len} tokens)")

    # Prefill with PyTorch
    position_ids = torch.arange(prompt_len, device=DEVICE)
    prefill_inp = BatchState(input_ids=input_ids, position_ids=position_ids)
    prefill_output = model(prefill_inp)
    assert prefill_output.output_ids is not None
    new_input_token = prefill_output.output_ids[:, -1:]

    # Prepare output buffer; slot 0 = first generated token (from prefill)
    output_tokens = torch.zeros(1, NTOK, device=DEVICE, dtype=torch.long)
    output_tokens[:, 0] = new_input_token

    # Build schedule and tensorize
    seq_len = prompt_len + NTOK
    schedule_builder = make_schedule_builder(SETTING)
    schedule = schedule_builder.build(model, seq_len=seq_len)
    assigned = assign_to_sms("rr", schedule=schedule)
    tensorize_instructions(schedule.globs, assigned)

    # Build MK interpreter and generator
    interpreter = make_mk_interpreter(SETTING, MK_DIR)
    gen = MK_Generator(model, interpreter, schedule)

    print(f"Generating {NTOK} tokens via MK...")
    gen.generate(output_tokens, prompt_len, ntok=NTOK - 1)
    torch.cuda.synchronize()
    print("Done.\n")

    decoded = tokenizer.decode(output_tokens[0].cpu().tolist(), skip_special_tokens=True)
    print("=== Response ===")
    print(decoded)


if __name__ == "__main__":
    main()
