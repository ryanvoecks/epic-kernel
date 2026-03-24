"""
Focused OProj diagnostics for Mixtral.

Goal:
- gather high-signal evidence for OProj mismatch without changing kernels.
- compare PyVM vs MK at stop_after_op='oproj'.
- quantify whether errors are structured by output block modulo pipeline stages.
- compare outputs against direct reference formulas using Py/MK attn_out inputs.

Usage:
  cd /home/of222/epic-kernel
  uv run python demos/mixtral/debug_oproj_issue.py
  uv run python demos/mixtral/debug_oproj_issue.py --rope identity
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


def tstats(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, float, float]:
    af = a.float()
    bf = b.float()
    ad = (af - bf).abs()
    rd = 2 * ad / (af.abs() + bf.abs() + 1e-6)
    return (
        float(ad.max().item()),
        float(ad.mean().item()),
        float(rd.mean().item()),
        float(rd.max().item()),
    )


def build_pair(rope_mode: str):
    model = base.FakeModel()
    if rope_mode == "identity":
        model.model.rope_cos.fill_(1.0)
        model.model.rope_sin.fill_(0.0)
    elif rope_mode == "zero":
        model.model.rope_cos.zero_()
        model.model.rope_sin.zero_()
    else:
        raise ValueError(f"unsupported rope mode: {rope_mode}")

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

    hidden_before = seed_h.clone().float()

    nodes_py, end_py = make_dag(gpy, stop_after_op="oproj", layer_limit=1)
    sched_py = Schedule(gpy, nodes_py, end_py)
    assigned = assign_to_sms("rr", schedule=sched_py)
    tensorize_instructions(gpy, assigned)
    gpy.barriers.fill_(0)

    nodes_mk, end_mk = make_dag(gmk, stop_after_op="oproj", layer_limit=1)
    sched_mk = Schedule(gmk, nodes_mk, end_mk)
    tensorize_instructions(gmk, assigned)
    gmk.barriers.fill_(0)

    pyvm = PyVM_Interpreter(INSTRUCTION_TO_SOLVER)
    pyvm.interpret(gpy, sched_py.get_linear_instructions())

    mk_func = importlib.import_module(base.MK_MODULE_NAME).mk_mixtral
    interpret_with_mk(gmk, mk_func)
    torch.cuda.synchronize()

    return gpy, gmk, hidden_before


def direct_ref(hidden_before: torch.Tensor, o_w: torch.Tensor, attn_out: torch.Tensor) -> torch.Tensor:
    # o_w: [hidden, hidden], attn_out: [hidden]
    return (o_w.float() @ attn_out.float()) + hidden_before.float()


def direct_ref_bf16_path(hidden_before: torch.Tensor, o_w: torch.Tensor, attn_out: torch.Tensor) -> torch.Tensor:
    # rough bf16 emulation for sensitivity checks
    prod = (o_w.to(torch.bfloat16) @ attn_out.to(torch.bfloat16)).to(torch.float32)
    out = prod + hidden_before.to(torch.bfloat16).to(torch.float32)
    return out


def print_block_stats(name: str, a: torch.Tensor, b: torch.Tensor, block: int = 16) -> None:
    af = a.float().view(-1, block)
    bf = b.float().view(-1, block)
    ad = (af - bf).abs()
    print(f"\n{name} per-block top-8 max abs diff:")
    tops = torch.topk(ad.max(dim=1).values, k=min(8, ad.shape[0]))
    for rank, (v, idx) in enumerate(zip(tops.values.tolist(), tops.indices.tolist()), 1):
        mean_ad = float(ad[idx].mean().item())
        print(f"  {rank:>2d}. block={idx:>2d} max_adiff={v:.6f} mean_adiff={mean_ad:.6f} mod3={idx % 3}")

    print("\nblock error by (block_idx % 3):")
    for m in range(3):
        rows = ad[torch.arange(ad.shape[0], device=ad.device) % 3 == m]
        if rows.numel() == 0:
            continue
        print(
            f"  mod3={m}:"
            f" mean_adiff={float(rows.mean()):.6f}"
            f" max_adiff={float(rows.max()):.6f}"
            f" p95_adiff={float(torch.quantile(rows.flatten(), 0.95)):.6f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rope", choices=["zero", "identity"], default="zero")
    args = parser.parse_args()

    gpy, gmk, hidden_before = build_pair(args.rope)

    print(f"rope_mode={args.rope}")

    # Upstream parity at OProj entry
    qkv_o = tstats(gpy.attn_out, gmk.attn_out)
    print(
        "attn_out(py vs mk):"
        f" max_adiff={qkv_o[0]:.6f} mean_adiff={qkv_o[1]:.6f}"
        f" mean_rdiff={qkv_o[2]:.6f} max_rdiff={qkv_o[3]:.6f}"
    )

    # Final OProj output parity
    oproj_s = tstats(gpy.hidden_states, gmk.hidden_states)
    print(
        "hidden_states_after_oproj(py vs mk):"
        f" max_adiff={oproj_s[0]:.6f} mean_adiff={oproj_s[1]:.6f}"
        f" mean_rdiff={oproj_s[2]:.6f} max_rdiff={oproj_s[3]:.6f}"
    )

    # Compare OProj contribution only
    py_contrib = gpy.hidden_states.float() - hidden_before
    mk_contrib = gmk.hidden_states.float() - hidden_before
    contrib_s = tstats(py_contrib, mk_contrib)
    print(
        "oproj_contrib(py vs mk):"
        f" max_adiff={contrib_s[0]:.6f} mean_adiff={contrib_s[1]:.6f}"
        f" mean_rdiff={contrib_s[2]:.6f} max_rdiff={contrib_s[3]:.6f}"
    )

    # Direct references
    o_w = gpy.o_proj_weights[0]
    ref_py_input = direct_ref(hidden_before, o_w, gpy.attn_out)
    ref_mk_input = direct_ref(hidden_before, o_w, gmk.attn_out)
    ref_bf16_mk_input = direct_ref_bf16_path(hidden_before, o_w, gmk.attn_out)

    s_ref_py_mk = tstats(ref_py_input, gmk.hidden_states)
    s_ref_mk_mk = tstats(ref_mk_input, gmk.hidden_states)
    s_ref_bf16_mk = tstats(ref_bf16_mk_input, gmk.hidden_states)

    print(
        "mk_hidden vs direct_ref(py_attn_out):"
        f" max_adiff={s_ref_py_mk[0]:.6f} mean_adiff={s_ref_py_mk[1]:.6f}"
        f" mean_rdiff={s_ref_py_mk[2]:.6f} max_rdiff={s_ref_py_mk[3]:.6f}"
    )
    print(
        "mk_hidden vs direct_ref(mk_attn_out):"
        f" max_adiff={s_ref_mk_mk[0]:.6f} mean_adiff={s_ref_mk_mk[1]:.6f}"
        f" mean_rdiff={s_ref_mk_mk[2]:.6f} max_rdiff={s_ref_mk_mk[3]:.6f}"
    )
    print(
        "mk_hidden vs direct_ref_bf16(mk_attn_out):"
        f" max_adiff={s_ref_bf16_mk[0]:.6f} mean_adiff={s_ref_bf16_mk[1]:.6f}"
        f" mean_rdiff={s_ref_bf16_mk[2]:.6f} max_rdiff={s_ref_bf16_mk[3]:.6f}"
    )

    print_block_stats("hidden_states_after_oproj", gpy.hidden_states, gmk.hidden_states)


if __name__ == "__main__":
    main()
