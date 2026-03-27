"""
Minimal diff test: random weights, PyVM vs MK for one Mixtral layer.

Run with:
  cd /home/of222/epic-kernel
  uv run python demos/mixtral/test_mixtral_mk.py
"""
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn.init import normal_

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "build"))

from megakernels.mixtral.instructions import MixtralGlobals
from megakernels.mixtral.python_vm import INSTRUCTION_TO_SOLVER
from megakernels.mixtral.scheduler import make_globals, make_dag
from megakernels.mixtral.mk import interpret_with_mk
from megakernels.python_vm import PyVM_Interpreter
from megakernels.scheduler import assign_to_sms, tensorize_instructions

# ---------------------------------------------------------------------------
# Small-test dimensions. Keeps GPU memory requirements to a few MB so tests
# run on a busy machine.
# ---------------------------------------------------------------------------
NUM_LAYERS        = 2       # run 2 layers to test cross-layer dependency
HIDDEN_DIM        = 512
INTERMEDIATE_DIM  = 1024
HEAD_DIM          = 64
NUM_ATTN_HEADS    = 8
NUM_KV_HEADS      = 2
NUM_EXPERTS       = 4
NUM_EXPERTS_TOK   = 2
VOCAB_SIZE        = 256
MAX_SEQ_LEN       = 64
SEQ_LEN           = 16  # pos_id = SEQ_LEN - 1

DEVICE = "cuda"
DTYPE  = torch.bfloat16

MK_MODULE_NAME = "mk_mixtral_small"


@dataclass
class FakeConfig:
    num_hidden_layers: int   = NUM_LAYERS
    hidden_size: int         = HIDDEN_DIM
    intermediate_size: int   = INTERMEDIATE_DIM
    num_attention_heads: int = NUM_ATTN_HEADS
    num_key_value_heads: int = NUM_KV_HEADS
    num_local_experts: int   = NUM_EXPERTS
    num_experts_per_tok: int = NUM_EXPERTS_TOK
    vocab_size: int          = VOCAB_SIZE
    rms_norm_eps: float      = 1e-5


@dataclass
class FakeStackedParams:
    qkv_proj:      torch.Tensor
    attn_ln_weight: torch.Tensor
    o_proj:        torch.Tensor
    ffn_ln_weight: torch.Tensor
    router_weight: torch.Tensor
    expert_gate:   torch.Tensor
    expert_up:     torch.Tensor
    expert_down:   torch.Tensor


@dataclass
class FakeLMHead:
    input_norm: object
    lm_head: object

    def __init__(self, device, dtype):
        class _W:
            def __init__(self):
                self.weight = None
        self.input_norm = _W()
        self.lm_head = _W()
        self.input_norm.weight = torch.zeros(HIDDEN_DIM, device=device, dtype=dtype)
        self.lm_head.weight = torch.zeros(VOCAB_SIZE, HIDDEN_DIM, device=device, dtype=dtype)
        normal_(self.input_norm.weight)
        normal_(self.lm_head.weight)


class FakeModel:
    def __init__(self):
        self.config = FakeConfig()
        self.device = DEVICE
        self.dtype = DTYPE

        def r(*shape):
            t = torch.zeros(*shape, device=DEVICE, dtype=DTYPE)
            normal_(t)
            return t

        qkv_out = (NUM_ATTN_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM
        self.stacked_params = FakeStackedParams(
            qkv_proj      = r(NUM_LAYERS, qkv_out, HIDDEN_DIM),
            attn_ln_weight= r(NUM_LAYERS, HIDDEN_DIM),
            o_proj        = r(NUM_LAYERS, HIDDEN_DIM, NUM_ATTN_HEADS * HEAD_DIM),
            ffn_ln_weight = r(NUM_LAYERS, HIDDEN_DIM),
            router_weight = r(NUM_LAYERS, NUM_EXPERTS, HIDDEN_DIM),
            expert_gate   = r(NUM_LAYERS, NUM_EXPERTS, INTERMEDIATE_DIM, HIDDEN_DIM),
            expert_up     = r(NUM_LAYERS, NUM_EXPERTS, INTERMEDIATE_DIM, HIDDEN_DIM),
            expert_down   = r(NUM_LAYERS, NUM_EXPERTS, HIDDEN_DIM, INTERMEDIATE_DIM),
        )
        self.lm_head = FakeLMHead(DEVICE, DTYPE)

        max_attn_partitions = 148  # B200 SM count
        self.stacked_kv_cache = [
            torch.zeros(NUM_LAYERS, 1, MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE),
            torch.zeros(NUM_LAYERS, 1, MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE),
        ]

        cos_sin = torch.zeros(MAX_SEQ_LEN, HEAD_DIM, device=DEVICE, dtype=torch.float32)

        class _Model:
            def __init__(self, cs):
                self.rope_cos = cs
                self.rope_sin = cs.clone()
        self.model = _Model(cos_sin)


def build_globs(model, seq_len):
    return make_globals(model, seq_len=seq_len)


def main():
    torch.manual_seed(42)
    model = FakeModel()

    seq_len = SEQ_LEN
    pos_id = seq_len - 1

    # Build two independent globals for PyVM and MK
    gpy = build_globs(model, seq_len)
    gmk = build_globs(model, seq_len)

    gpy.pos_id = pos_id
    gmk.pos_id = pos_id

    # Initialize hidden states identically
    normal_(gpy.hidden_states)
    gmk.hidden_states.copy_(gpy.hidden_states)

    # Initialize KV cache with some data up to seq_len
    normal_(gpy.k_cache[:, :, :seq_len])
    normal_(gpy.v_cache[:, :, :seq_len])
    gmk.k_cache = gpy.k_cache.clone()
    gmk.v_cache = gpy.v_cache.clone()

    # Build schedule
    nodes, end_node = make_dag(gpy, layer_limit=NUM_LAYERS)

    from megakernels.scheduler import Schedule
    schedule_py = Schedule(gpy, nodes, end_node)
    schedule_mk = Schedule(gmk, nodes, end_node)

    assigned = assign_to_sms(mode="rr", schedule=schedule_mk)

    tensorize_instructions(gpy, assigned)
    tensorize_instructions(gmk, assigned)

    gpy.barriers.fill_(0)
    gmk.barriers.fill_(0)

    # Run PyVM
    pyvm = PyVM_Interpreter(INSTRUCTION_TO_SOLVER)
    linear_instructions = schedule_py.get_linear_instructions()
    print(f"Running PyVM with {len(linear_instructions)} instructions...")
    pyvm.interpret(gpy, linear_instructions)
    torch.cuda.synchronize()
    print("PyVM done.")

    # Load MK module
    build_dir = Path(__file__).parent.parent.parent / "build"
    sys.path.insert(0, str(build_dir))
    import importlib
    mk_mod = importlib.import_module(MK_MODULE_NAME)
    mk_func = mk_mod.mk_mixtral  # bound function is always named "mk_mixtral"

    print("Running MK...")
    try:
        interpret_with_mk(gmk, mk_func)
        torch.cuda.synchronize()
        print("MK done.")
    except Exception as e:
        print(f"MK failed: {e}")
        raise

    # Compare outputs
    print("\n=== Diff Results ===")
    gpy.diff(gmk)


if __name__ == "__main__":
    main()
