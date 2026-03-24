"""
Focused QKV diagnostics for Mixtral small-test kernel parity.

This script helps distinguish true QKV math bugs from masked comparisons when
RoPE tables are zero (which can hide Q/K mismatches).

Usage:
  cd /home/of222/epic-kernel
  uv run python demos/mixtral/debug_qkv_precision.py
  uv run python demos/mixtral/debug_qkv_precision.py --rope identity
  uv run python demos/mixtral/debug_qkv_precision.py --token 15
"""

from __future__ import annotations

import argparse
import importlib

import torch
from torch.nn.init import normal_

import demos.mixtral.debug_mixtral_mk as base
from megakernels.mixtral.mk import interpret_with_mk
from megakernels.mixtral.python_vm import INSTRUCTION_TO_SOLVER
from megakernels.mixtral.scheduler import make_dag, make_globals
from megakernels.python_vm import PyVM_Interpreter
from megakernels.scheduler import Schedule, assign_to_sms, tensorize_instructions


def tstats(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, float]:
    af = a.float()
    bf = b.float()
    adiff = (af - bf).abs()
    rdiff = 2 * adiff / (af.abs() + bf.abs() + 1e-6)
    return float(adiff.max().item()), float(rdiff.mean().item()), float(rdiff.max().item())


def build_pair(rope_mode: str):
    model = base.FakeModel()
    if rope_mode == "identity":
        model.model.rope_cos.fill_(1.0)
        model.model.rope_sin.fill_(0.0)
    elif rope_mode == "zero":
        model.model.rope_cos.zero_()
        model.model.rope_sin.zero_()
    else:
        raise ValueError(f"unsupported rope_mode={rope_mode}")

    mk_func = importlib.import_module(base.MK_MODULE_NAME).mk_mixtral

    torch.manual_seed(base.DEFAULT_SEED)
    seed_h = torch.zeros(base.HIDDEN_DIM, device=base.DEVICE, dtype=base.DTYPE)
    normal_(seed_h)
    seed_k = torch.zeros_like(model.stacked_kv_cache[0])
    seed_v = torch.zeros_like(model.stacked_kv_cache[1])
    normal_(seed_k[:, :, : base.SEQ_LEN])
    normal_(seed_v[:, :, : base.SEQ_LEN])

    # PyVM globals
    model.stacked_kv_cache[0].copy_(seed_k)
    model.stacked_kv_cache[1].copy_(seed_v)
    gpy = make_globals(model, seq_len=base.SEQ_LEN)
    gpy.pos_id = base.SEQ_LEN - 1
    gpy.hidden_states.copy_(seed_h)
    gpy.k_cache = seed_k.clone()
    gpy.v_cache = seed_v.clone()

    # MK globals
    model.stacked_kv_cache[0].copy_(seed_k)
    model.stacked_kv_cache[1].copy_(seed_v)
    gmk = make_globals(model, seq_len=base.SEQ_LEN)
    gmk.pos_id = base.SEQ_LEN - 1
    gmk.hidden_states.copy_(seed_h)

    nodes_py, end_py = make_dag(gpy, stop_after_op="qkv", layer_limit=1)
    sched_py = Schedule(gpy, nodes_py, end_py)
    assigned = assign_to_sms("rr", schedule=sched_py)
    tensorize_instructions(gpy, assigned)
    gpy.barriers.fill_(0)

    nodes_mk, end_mk = make_dag(gmk, stop_after_op="qkv", layer_limit=1)
    sched_mk = Schedule(gmk, nodes_mk, end_mk)
    tensorize_instructions(gmk, assigned)
    gmk.barriers.fill_(0)

    PyVM_Interpreter(INSTRUCTION_TO_SOLVER).interpret(gpy, sched_py.get_linear_instructions())
    interpret_with_mk(gmk, mk_func)
    torch.cuda.synchronize()

    return gpy, gmk


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rope", choices=["zero", "identity"], default="zero")
    p.add_argument("--token", type=int, default=base.SEQ_LEN - 1)
    args = p.parse_args()

    gpy, gmk = build_pair(args.rope)

    print(f"rope_mode={args.rope}")

    for name in ["post_ln_rope_q", "k_cache", "v_cache"]:
        a = getattr(gpy, name)
        b = getattr(gmk, name)
        mad, mrd, xrd = tstats(a, b)
        print(f"{name:>18s}: max_adiff={mad:.6f} mean_rdiff={mrd:.6f} max_rdiff={xrd:.6f}")

    t = args.token
    py_k = gpy.k_cache[0, 0, t].float()
    mk_k = gmk.k_cache[0, 0, t].float()
    py_v = gpy.v_cache[0, 0, t].float()
    mk_v = gmk.v_cache[0, 0, t].float()

    mad, mrd, xrd = tstats(py_k, mk_k)
    print(f"k_cache[token={t}]: max_adiff={mad:.6f} mean_rdiff={mrd:.6f} max_rdiff={xrd:.6f}")
    mad, mrd, xrd = tstats(py_v, mk_v)
    print(f"v_cache[token={t}]: max_adiff={mad:.6f} mean_rdiff={mrd:.6f} max_rdiff={xrd:.6f}")


if __name__ == "__main__":
    main()
