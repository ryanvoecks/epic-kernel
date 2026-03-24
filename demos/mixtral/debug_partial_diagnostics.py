"""
Detailed diagnostics for Mixtral partial attention mismatch.

This script isolates where divergence begins by running controlled case combinations:
1) Normal schedule (usually skip mode for small test).
2) One-partition schedule with runtime skip branch forced ON/OFF.
3) Forced multi-partition schedule (non-skip path).

It reports:
- PyVM vs MK metrics for attn_out / attn_out_intermediates / attn_lse_intermediates.
- MK skip-output vs MK no-skip-intermediate parity (same one-partition schedule).
- Per-head and per-column mismatch masks for precise failure localization.

Run:
  cd /home/of222/epic-kernel
  uv run python demos/mixtral/debug_partial_diagnostics.py
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass

import torch
from torch.nn.init import normal_

import demos.mixtral.debug_mixtral_mk as base
import megakernels.mixtral.scheduler as sched_mod
from megakernels.mixtral.mk import interpret_with_mk
from megakernels.mixtral.python_vm import INSTRUCTION_TO_SOLVER
from megakernels.mixtral.scheduler import make_dag, make_globals
from megakernels.python_vm import PyVM_Interpreter
from megakernels.scheduler import Schedule, assign_to_sms, tensorize_instructions


@dataclass
class RunResult:
    tag: str
    gpy: object
    gmk: object


def tstats(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, float]:
    af = a.float()
    bf = b.float()
    adiff = (af - bf).abs()
    rdiff = 2 * adiff / (af.abs() + bf.abs() + 1e-6)
    return float(adiff.max().item()), float(rdiff.mean().item()), float(rdiff.max().item())


def run_case(
    tag: str,
    force_partitions: int | None,
    force_skip_runtime: bool | None,
) -> RunResult:
    model = base.FakeModel()
    mk_func = importlib.import_module(base.MK_MODULE_NAME).mk_mixtral

    orig_pick = sched_mod.pick_num_attention_partitions
    if force_partitions is not None:
        sched_mod.pick_num_attention_partitions = lambda *args, **kwargs: force_partitions

    try:
        torch.manual_seed(42)
        seed_h = torch.zeros(base.HIDDEN_DIM, device=base.DEVICE, dtype=base.DTYPE)
        normal_(seed_h)

        seed_k = torch.zeros_like(model.stacked_kv_cache[0])
        seed_v = torch.zeros_like(model.stacked_kv_cache[1])
        normal_(seed_k[:, :, : base.SEQ_LEN])
        normal_(seed_v[:, :, : base.SEQ_LEN])

        def build_globals(is_pyvm: bool) -> tuple[object, Schedule]:
            model.stacked_kv_cache[0].copy_(seed_k)
            model.stacked_kv_cache[1].copy_(seed_v)
            g = make_globals(model, seq_len=base.SEQ_LEN)
            g.pos_id = base.SEQ_LEN - 1
            g.hidden_states.copy_(seed_h)
            # Keep the PyVM reference independent from model-owned cache tensors.
            if is_pyvm:
                g.k_cache = seed_k.clone()
                g.v_cache = seed_v.clone()
            nodes, end_node = make_dag(g, stop_after_op="partial", layer_limit=1)
            sched = Schedule(g, nodes, end_node)
            assigned = assign_to_sms("rr", schedule=sched)
            tensorize_instructions(g, assigned)
            g.barriers.fill_(0)
            if force_skip_runtime is not None:
                g.skip_attn_reduction = force_skip_runtime
            return g, sched

        gpy, sched_py = build_globals(is_pyvm=True)
        PyVM_Interpreter(INSTRUCTION_TO_SOLVER).interpret(gpy, sched_py.get_linear_instructions())

        gmk, _ = build_globals(is_pyvm=False)
        interpret_with_mk(gmk, mk_func)
        torch.cuda.synchronize()

        return RunResult(tag=tag, gpy=gpy, gmk=gmk)
    finally:
        sched_mod.pick_num_attention_partitions = orig_pick


def print_py_vs_mk_stats(run: RunResult) -> None:
    print(f"\n=== {run.tag} ===")
    print(
        "flags:"
        f" py.skip_attn_reduction={bool(run.gpy.skip_attn_reduction)}"
        f" mk.skip_attn_reduction={bool(run.gmk.skip_attn_reduction)}"
    )

    for name in ["attn_out", "attn_out_intermediates", "attn_lse_intermediates"]:
        a = getattr(run.gpy, name)
        b = getattr(run.gmk, name)
        mad, mrd, xrd = tstats(a, b)
        print(
            f"{name:>24s}:"
            f" max_adiff={mad:.6f}"
            f" mean_rdiff={mrd:.6f}"
            f" max_rdiff={xrd:.6f}"
        )


def print_skip_vs_noskip_mk(skip_run: RunResult, noskip_run: RunResult) -> None:
    print("\n=== mk_skip_vs_mk_noskip_same_schedule ===")

    mk_skip = skip_run.gmk.attn_out.view(base.NUM_ATTN_HEADS, base.HEAD_DIM).float()
    mk_noskip_inter = (
        noskip_run.gmk.attn_out_intermediates[:, 0, :].to(torch.bfloat16).float()
    )

    mad, mrd, xrd = tstats(mk_skip, mk_noskip_inter)
    print(
        "skip_attn_out vs noskip_intermediate(partial=0):"
        f" max_adiff={mad:.6f} mean_rdiff={mrd:.6f} max_rdiff={xrd:.6f}"
    )

    for h in range(base.NUM_ATTN_HEADS):
        adiff = (mk_skip[h] - mk_noskip_inter[h]).abs()
        rdiff = 2 * adiff / (mk_skip[h].abs() + mk_noskip_inter[h].abs() + 1e-6)
        bad_cols = (adiff > 1e-4).nonzero(as_tuple=False).flatten().tolist()
        print(
            f"head {h}:"
            f" max_adiff={float(adiff.max()):.6f}"
            f" mean_rdiff={float(rdiff.mean()):.6f}"
            f" bad_col_count={len(bad_cols)}"
            f" bad_cols_first16={bad_cols[:16]}"
        )


def print_producer_consistency(noskip_run: RunResult) -> None:
    print("\n=== producer_consistency_non_skip ===")
    py_inter = noskip_run.gpy.attn_out_intermediates[:, 0, :].float()
    mk_inter = noskip_run.gmk.attn_out_intermediates[:, 0, :].float()

    mad, mrd, xrd = tstats(py_inter, mk_inter)
    print(
        "py_vs_mk attn_out_intermediates (active partial):"
        f" max_adiff={mad:.6f} mean_rdiff={mrd:.6f} max_rdiff={xrd:.6f}"
    )

    for h in range(base.NUM_ATTN_HEADS):
        adiff = (py_inter[h] - mk_inter[h]).abs()
        rdiff = 2 * adiff / (py_inter[h].abs() + mk_inter[h].abs() + 1e-6)
        print(
            f"head {h}:"
            f" max_adiff={float(adiff.max()):.6f}"
            f" mean_rdiff={float(rdiff.mean()):.6f}"
            f" py0_4={[round(float(v), 4) for v in py_inter[h, :4]]}"
            f" mk0_4={[round(float(v), 4) for v in mk_inter[h, :4]]}"
        )


def main() -> None:
    # A: normal schedule behavior
    normal = run_case(
        tag="normal_schedule",
        force_partitions=None,
        force_skip_runtime=None,
    )

    # B/C: one-partition schedule, runtime branch toggled
    one_skip = run_case(
        tag="one_partition_force_skip_runtime_true",
        force_partitions=None,
        force_skip_runtime=True,
    )
    one_noskip = run_case(
        tag="one_partition_force_skip_runtime_false",
        force_partitions=None,
        force_skip_runtime=False,
    )

    # D: forced non-skip via scheduler partitions
    forced_non_skip = run_case(
        tag="forced_partitions_2",
        force_partitions=2,
        force_skip_runtime=None,
    )

    print_py_vs_mk_stats(normal)
    print_py_vs_mk_stats(one_skip)
    print_py_vs_mk_stats(one_noskip)
    print_py_vs_mk_stats(forced_non_skip)

    print_skip_vs_noskip_mk(one_skip, one_noskip)
    print_producer_consistency(one_noskip)


if __name__ == "__main__":
    main()
