"""
Stage-by-stage debug script: PyVM (reference) vs MK (CUDA) for each Mixtral opcode.

For each stage, fresh identical globals are built, PyVM and MK both run only
the instructions up to that stage, and their output buffers are compared.
This narrows the failure to the first stage where outputs diverge.

Run with:
  cd /home/of222/epic-kernel
  uv run python demos/mixtral/debug_mixtral_mk.py
  uv run python demos/mixtral/debug_mixtral_mk.py --stage oproj
  uv run python demos/mixtral/debug_mixtral_mk.py --shapes-only
"""
import argparse
import importlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.nn.init import normal_

BUILD_DIR = Path(__file__).parent.parent.parent / "build"
sys.path.insert(0, str(BUILD_DIR))

from megakernels.mixtral.instructions import MixtralGlobals
from megakernels.mixtral.python_vm import INSTRUCTION_TO_SOLVER
from megakernels.mixtral.scheduler import make_globals, make_dag
from megakernels.mixtral.mk import interpret_with_mk
from megakernels.python_vm import PyVM_Interpreter
from megakernels.scheduler import Schedule, assign_to_sms, tensorize_instructions

# ---------------------------------------------------------------------------
# Dimensions — must match full-scale Mixtral in mixtral.cuh
# ---------------------------------------------------------------------------
NUM_LAYERS       = 1   # single layer isolates without cross-layer barrier deps
HIDDEN_DIM       = 4096
INTERMEDIATE_DIM = 14336
HEAD_DIM         = 128
NUM_ATTN_HEADS   = 32
NUM_KV_HEADS     = 8
NUM_EXPERTS      = 8
NUM_EXPERTS_TOK  = 2
VOCAB_SIZE       = 32000
MAX_SEQ_LEN      = 64
SEQ_LEN          = 16

DEVICE = "cuda"
DTYPE  = torch.bfloat16
MK_MODULE_NAME = "mk_mixtral"
DEFAULT_SEED = 42

ATOL = 1.5    # full-scale bf16 values are larger; 1 ULP at magnitude ~256 = 0.5, at ~512 = 1.0
RTOL = 5e-2

# Full-scale uses the real TMA pipeline path (not the SMALL_TEST scalar fallback).
# Tolerances may need adjustment based on bf16 accumulation variance.
STAGE_BUFFER_ATOL: dict[tuple[str, str], float] = {
    # OProj accumulates bfloat16 store_add_async from multiple SMs.
    # At hidden_states magnitudes ~600, 1 bfloat16 ULP = 4.  Allow 2 ULPs.
    ("oproj", "hidden_states"):              8.0,
    # At full scale, expert_silu_out magnitudes reach ~16000 where 1 bf16 ULP = 128.
    # Allow 4 ULPs of slack for multi-SM accumulation + bf16 rounding.
    ("rms_router_upgate", "expert_silu_out"): 512.0,
    # Downproj now reduces the full intermediate_dim in one instruction.
    # Only 2 store_add_async per element (one per expert).  At full-scale
    # magnitudes (~300K), bf16 ULP = 2048.  Allow 4 ULPs of slack.
    # Downproj: 2 store_add_async per element (one per expert) + bf16 rounding
    # from float32 accumulation. At magnitudes ~370K, bf16 ULP = 2048-4096.
    # Allow ~5 ULPs.
    ("downproj_fused", "hidden_states"):     16384.0,
    # The full-stage hidden_states check sees the same downproj error.
    ("full", "hidden_states"):               16384.0,
}

# Ordered list of testable stages
ALL_STAGES = ["qkv", "partial", "reduction", "oproj", "rms_router_upgate", "downproj_fused", "full"]

# stop_after_op argument for make_dag (None = full)
STAGE_STOP = {
    "qkv":                "qkv",
    "partial":            "partial",
    "reduction":          "reduction",
    "oproj":              "oproj",
    "rms_router_upgate":  "rms_router_upgate",
    "downproj_fused":     "downproj_fused",
    "full":               None,
}

# Buffers to compare at each stage (attr names on MixtralGlobals)
STAGE_BUFFERS = {
    "qkv":                ["post_ln_rope_q", "k_cache", "v_cache"],
    "partial":            ["attn_out", "attn_out_intermediates", "attn_lse_intermediates"],
    "reduction":          ["attn_out"],
    "oproj":              ["hidden_states"],
    "rms_router_upgate":  ["selected_expert_indices", "selected_expert_scores", "router_normed_hidden", "expert_silu_out"],
    "downproj_fused":     ["hidden_states"],
    "full":               ["logits", "hidden_states"],
}


# ---------------------------------------------------------------------------
# Fake model matching test_mixtral_mk.py
# ---------------------------------------------------------------------------

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
    qkv_proj:      Tensor
    attn_ln_weight: Tensor
    o_proj:        Tensor
    ffn_ln_weight: Tensor
    router_weight: Tensor
    expert_gate:   Tensor
    expert_up:     Tensor
    expert_down:   Tensor


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
            qkv_proj       = r(NUM_LAYERS, qkv_out, HIDDEN_DIM),
            attn_ln_weight = r(NUM_LAYERS, HIDDEN_DIM),
            o_proj         = r(NUM_LAYERS, HIDDEN_DIM, NUM_ATTN_HEADS * HEAD_DIM),
            ffn_ln_weight  = r(NUM_LAYERS, HIDDEN_DIM),
            router_weight  = r(NUM_LAYERS, NUM_EXPERTS, HIDDEN_DIM),
            expert_gate    = r(NUM_LAYERS, NUM_EXPERTS, INTERMEDIATE_DIM, HIDDEN_DIM),
            expert_up      = r(NUM_LAYERS, NUM_EXPERTS, INTERMEDIATE_DIM, HIDDEN_DIM),
            expert_down    = r(NUM_LAYERS, NUM_EXPERTS, HIDDEN_DIM, INTERMEDIATE_DIM),
        )
        self.lm_head = FakeLMHead(DEVICE, DTYPE)
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

def ok(msg):    print(f"  {GREEN}PASS{RESET} {msg}")
def fail(msg):  print(f"  {RED}FAIL{RESET} {msg}")
def warn(msg):  print(f"  {YELLOW}WARN{RESET} {msg}")


def compare_tensor(py_val: Tensor, mk_val: Tensor, name: str, atol: float = ATOL) -> bool:
    """Compare two tensors; return True if they agree within tolerance."""
    if py_val.dtype in (torch.int32, torch.int64):
        match = torch.all(py_val == mk_val).item()
        if match:
            ok(f"{name}  (exact int match, shape {list(py_val.shape)})")
        else:
            fail(f"{name}  py={py_val.tolist()}  mk={mk_val.tolist()}")
        return match

    pf = py_val.float()
    mf = mk_val.float()
    adiff = (pf - mf).abs()
    rdiff = 2 * adiff / (pf.abs() + mf.abs() + 1e-6)
    max_adiff = adiff.max().item()
    mean_rdiff = rdiff.mean().item()
    max_rdiff  = rdiff.max().item()

    # Use ATOL only — RTOL causes false failures on near-zero values (large relative
    # error even when absolute difference is negligible, e.g. expert scores ≈ 0.0001).
    passed = max_adiff <= atol
    tag = ok if passed else fail
    tag(
        f"{name:<35s}  shape={list(py_val.shape)}"
        f"  max_adiff={max_adiff:.4f}  mean_rdiff={mean_rdiff:.4f}  max_rdiff={max_rdiff:.4f}"
        f"  (atol={atol:.4f})"
    )
    if not passed:
        # Show first 8 values for quick inspection
        flat_py = pf.flatten()[:8].tolist()
        flat_mk = mf.flatten()[:8].tolist()
        print(f"         py[0:8] = {[f'{v:.4f}' for v in flat_py]}")
        print(f"         mk[0:8] = {[f'{v:.4f}' for v in flat_mk]}")
        # Show max-diff location
        max_idx = adiff.flatten().argmax().item()
        unraveled = list(divmod(max_idx, py_val.shape[-1]))
        print(f"         max_adiff at flat={max_idx} -> {unraveled}: py={pf.flatten()[max_idx]:.4f}  mk={mf.flatten()[max_idx]:.4f}")
        # Show non-zero elements near max for context
        nz_py = (pf.abs() > 0.01).nonzero(as_tuple=False)
        nz_mk = (mf.abs() > 0.01).nonzero(as_tuple=False)
        print(f"         py non-zero count: {len(nz_py)}  mk non-zero count: {len(nz_mk)}")
    return passed


def load_mk_func():
    mk_mod = importlib.import_module(MK_MODULE_NAME)
    return mk_mod.mk_mixtral


def build_globs(model: FakeModel, seed_hidden: Tensor | None = None) -> MixtralGlobals:
    g = make_globals(model, seq_len=SEQ_LEN)
    g.pos_id = SEQ_LEN - 1
    if seed_hidden is not None:
        g.hidden_states.copy_(seed_hidden)
    else:
        normal_(g.hidden_states)
    # Fill KV cache with realistic data up to pos_id
    normal_(g.k_cache[:, :, :SEQ_LEN])
    normal_(g.v_cache[:, :, :SEQ_LEN])
    return g


def build_schedule_and_tensorize(g: MixtralGlobals, stop_after_op: str | None) -> Schedule:
    nodes, end_node = make_dag(g, stop_after_op=stop_after_op, layer_limit=NUM_LAYERS)
    schedule = Schedule(g, nodes, end_node)
    assigned = assign_to_sms(mode="rr", schedule=schedule)
    tensorize_instructions(g, assigned)
    g.barriers.fill_(0)
    return schedule


# ---------------------------------------------------------------------------
# Shape report (quick sanity without running compute)
# ---------------------------------------------------------------------------

def print_tensor_shapes(g: MixtralGlobals):
    """Print every tensor's shape/dtype in the globals struct for binding verification."""
    from dataclasses import fields
    print("\n=== MixtralGlobals tensor shapes ===")
    for f in fields(g):
        v = getattr(g, f.name)
        if isinstance(v, Tensor):
            print(f"  {f.name:<35s}  {str(list(v.shape)):<30s}  {v.dtype}  contiguous={v.is_contiguous()}")
    # Also print flattened expert tensors as they appear in MK call
    ne = g.num_experts
    nl = g.num_hidden_layers
    print("\n--- Expert tensors after flatten (as passed to MK) ---")
    for attr, name in [
        ("expert_gate_weights", "expert_gate (flat)"),
        ("expert_up_weights",   "expert_up   (flat)"),
        ("expert_down_weights", "expert_down (flat)"),
    ]:
        t = getattr(g, attr)
        flat = t.reshape(nl * ne, *t.shape[2:]).contiguous()
        print(f"  {name:<35s}  {str(list(flat.shape)):<30s}  {flat.dtype}")
    rw = g.router_weights.reshape(nl * ne, -1).contiguous()
    print(f"  {'router_weights (flat)':<35s}  {str(list(rw.shape)):<30s}  {rw.dtype}")
    lm_norm = g.lm_head_norm_weights.unsqueeze(0)
    print(f"  {'lm_head_norm (unsqueeze)':<35s}  {str(list(lm_norm.shape)):<30s}  {lm_norm.dtype}")

    from einops import rearrange
    k4d = rearrange(g.k_cache, "l b t h d -> (l b) t h d")
    v4d = rearrange(g.v_cache, "l b t h d -> (l b) t h d")
    print(f"  {'k_cache (4D for CUDA)':<35s}  {str(list(k4d.shape)):<30s}  {k4d.dtype}")
    print(f"  {'v_cache (4D for CUDA)':<35s}  {str(list(v4d.shape)):<30s}  {v4d.dtype}")


# ---------------------------------------------------------------------------
# Per-stage test
# ---------------------------------------------------------------------------

def run_stage_test(stage: str, model: FakeModel, mk_func) -> bool:
    """
    Run PyVM and MK with instructions limited to `stage`, compare outputs.
    Returns True if all compared buffers match.
    """
    stop = STAGE_STOP[stage]
    print(f"\n{'='*60}")
    print(f"  Stage: {stage.upper()}  (stop_after_op={stop!r})")
    print(f"{'='*60}")

    # "reduction" stage is a no-op when skip_attn_reduction=True (seq too short for multi-partition)
    if stage == "reduction":
        from megakernels.mixtral.scheduler import pick_num_attention_partitions
        np = pick_num_attention_partitions(SEQ_LEN, 0, NUM_KV_HEADS, DEVICE)
        if np == 1:
            warn(f"Skipping 'reduction' stage: seq_len={SEQ_LEN} → num_partitions=1 (skip_attn_reduction=True)")
            return True

    torch.manual_seed(42)

    # Build identical initial hidden states
    seed_h = torch.zeros(HIDDEN_DIM, device=DEVICE, dtype=DTYPE)
    normal_(seed_h)
    seed_k = torch.zeros_like(model.stacked_kv_cache[0])
    seed_v = torch.zeros_like(model.stacked_kv_cache[1])
    normal_(seed_k[:, :, :SEQ_LEN])
    normal_(seed_v[:, :, :SEQ_LEN])

    # PyVM globals
    model.stacked_kv_cache[0].copy_(seed_k)
    model.stacked_kv_cache[1].copy_(seed_v)
    gpy = make_globals(model, seq_len=SEQ_LEN)
    gpy.pos_id = SEQ_LEN - 1
    gpy.hidden_states.copy_(seed_h)
    # Detach test globals from model-owned KV tensors so the MK run cannot
    # mutate the already-computed PyVM reference buffers.
    gpy.k_cache = seed_k.clone()
    gpy.v_cache = seed_v.clone()

    # MK globals (same model weights, same initial state)
    model.stacked_kv_cache[0].copy_(seed_k)
    model.stacked_kv_cache[1].copy_(seed_v)
    gmk = make_globals(model, seq_len=SEQ_LEN)
    gmk.pos_id = SEQ_LEN - 1
    gmk.hidden_states.copy_(seed_h)

    # Build schedule and tensorize (returns Schedule with DAG for linear ordering)
    sched_py = build_schedule_and_tensorize(gpy, stop)
    build_schedule_and_tensorize(gmk, stop)

    num_instrs = (gmk.instructions != 0).any(dim=-1).sum().item()
    print(f"  Instructions in schedule: {num_instrs}")

    # Run PyVM
    linear = sched_py.get_linear_instructions()
    pyvm = PyVM_Interpreter(INSTRUCTION_TO_SOLVER)
    try:
        pyvm.interpret(gpy, linear)
        torch.cuda.synchronize()
    except Exception as e:
        fail(f"PyVM crashed: {e}")
        return False

    # Run MK
    try:
        interpret_with_mk(gmk, mk_func)
        torch.cuda.synchronize()
    except Exception as e:
        fail(f"MK crashed: {e}")
        import traceback; traceback.print_exc()
        return False

    # Compare buffers
    buffers = STAGE_BUFFERS[stage]
    all_pass = True
    print(f"\n  Comparing buffers: {buffers}")
    for name in buffers:
        py_val = getattr(gpy, name, None)
        mk_val = getattr(gmk, name, None)
        if py_val is None or mk_val is None:
            warn(f"{name} not found on globals")
            continue
        if not isinstance(py_val, Tensor):
            print(f"  {name}: (not a tensor, skipping)")
            continue
        local_atol = STAGE_BUFFER_ATOL.get((stage, name), ATOL)
        passed = compare_tensor(py_val, mk_val, name, atol=local_atol)
        all_pass = all_pass and passed

    # Extra: check barriers match
    print(f"\n  Barriers (PyVM vs MK):")
    py_bar = gpy.barriers.cpu()
    mk_bar = gmk.barriers.cpu()
    bar_match = torch.all(py_bar == mk_bar).item()
    if bar_match:
        ok("barriers match")
    else:
        fail("barriers differ")
        diff_mask = (py_bar != mk_bar)
        diff_idx = diff_mask.nonzero(as_tuple=False)
        for idx in diff_idx[:16]:
            l, op, h = idx.tolist()
            fail(f"  barriers[layer={l}, opcode={op+1}, h={h}]: py={py_bar[l,op,h].item()}  mk={mk_bar[l,op,h].item()}")

    return all_pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=ALL_STAGES + ["all"], default="all",
        help="Which stage to test (default: all, stops at first failure)",
    )
    parser.add_argument(
        "--shapes-only", action="store_true",
        help="Just print tensor shapes; do not run compute",
    )
    parser.add_argument(
        "--no-stop", action="store_true",
        help="Run all stages even if one fails (default: stop at first failure)",
    )
    args = parser.parse_args()

    # Seed before model construction so random weights are stable across runs.
    torch.manual_seed(DEFAULT_SEED)
    model = FakeModel()

    if args.shapes_only:
        torch.manual_seed(DEFAULT_SEED)
        g = make_globals(model, seq_len=SEQ_LEN)
        print_tensor_shapes(g)
        return

    print(f"\nLoading MK module '{MK_MODULE_NAME}' from {BUILD_DIR}...")
    try:
        mk_func = load_mk_func()
        print("  Module loaded OK")
    except Exception as e:
        print(f"{RED}ERROR{RESET}: Failed to import {MK_MODULE_NAME}: {e}")
        print("  Build with: cd demos/mixtral && make (with MIXTRAL_SMALL_TEST defined)")
        sys.exit(1)

    stages = ALL_STAGES if args.stage == "all" else [args.stage]

    results: dict[str, bool] = {}
    for stage in stages:
        passed = run_stage_test(stage, model, mk_func)
        results[stage] = passed
        if not passed and not args.no_stop and args.stage == "all":
            print(f"\n{RED}>>> First failure at stage '{stage}'. Stopping.{RESET}")
            print("    The CUDA kernel for this opcode likely has a bug.")
            print("    Re-run with --no-stop to see all stages.")
            break

    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    for stage, passed in results.items():
        status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        print(f"  {stage:<20s}  {status}")

    if all(results.values()):
        print(f"\n{GREEN}All tested stages passed!{RESET}")
    else:
        failed = [s for s, p in results.items() if not p]
        print(f"\n{RED}Failed stages: {failed}{RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()
