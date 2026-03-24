"""
Focused debug for Mixtral partial-attention failure localization.

This script runs two controlled experiments:
1) Normal scheduling (expected skip mode for small test): compare final attn_out.
2) Forced non-skip (num_partitions=2): compare attn_out_intermediates/attn_lse_intermediates.

Interpretation:
- If forced non-skip intermediates already diverge, the bug is upstream of skip final-store.
- If forced non-skip intermediates match but skip attn_out diverges, bug is in skip output path.

Run:
  cd /home/of222/epic-kernel
  uv run python demos/mixtral/debug_partial_failure_point.py
"""

import importlib
from dataclasses import dataclass

import torch
from torch.nn.init import normal_

import demos.mixtral.debug_mixtral_mk as base
import megakernels.mixtral.scheduler as sched_mod
from megakernels.mixtral.scheduler import make_globals, make_dag
from megakernels.mixtral.python_vm import INSTRUCTION_TO_SOLVER
from megakernels.mixtral.mk import interpret_with_mk
from megakernels.python_vm import PyVM_Interpreter
from megakernels.scheduler import Schedule, assign_to_sms, tensorize_instructions


@dataclass
class CaseResult:
    case_name: str
    skip_attn_reduction: bool
    max_attn_out_adiff: float
    mean_attn_out_rdiff: float
    max_intermediate_adiff: float
    mean_intermediate_rdiff: float
    max_lse_adiff: float
    mean_lse_rdiff: float


def tensor_stats(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    af = a.float()
    bf = b.float()
    adiff = (af - bf).abs()
    rdiff = 2 * adiff / (af.abs() + bf.abs() + 1e-6)
    return float(adiff.max().item()), float(rdiff.mean().item())


def build_and_run(force_partitions: int | None) -> CaseResult:
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

        # PyVM state
        model.stacked_kv_cache[0].copy_(seed_k)
        model.stacked_kv_cache[1].copy_(seed_v)
        gpy = make_globals(model, seq_len=base.SEQ_LEN)
        gpy.pos_id = base.SEQ_LEN - 1
        gpy.hidden_states.copy_(seed_h)

        nodes_py, end_py = make_dag(gpy, stop_after_op="partial", layer_limit=1)
        sched_py = Schedule(gpy, nodes_py, end_py)
        assigned = assign_to_sms("rr", schedule=sched_py)
        tensorize_instructions(gpy, assigned)
        gpy.barriers.fill_(0)
        PyVM_Interpreter(INSTRUCTION_TO_SOLVER).interpret(gpy, sched_py.get_linear_instructions())

        # MK state
        model.stacked_kv_cache[0].copy_(seed_k)
        model.stacked_kv_cache[1].copy_(seed_v)
        gmk = make_globals(model, seq_len=base.SEQ_LEN)
        gmk.pos_id = base.SEQ_LEN - 1
        gmk.hidden_states.copy_(seed_h)

        nodes_mk, end_mk = make_dag(gmk, stop_after_op="partial", layer_limit=1)
        sched_mk = Schedule(gmk, nodes_mk, end_mk)
        tensorize_instructions(gmk, assigned)
        gmk.barriers.fill_(0)
        interpret_with_mk(gmk, mk_func)
        torch.cuda.synchronize()

        out_adiff, out_rdiff = tensor_stats(gpy.attn_out, gmk.attn_out)
        inter_adiff, inter_rdiff = tensor_stats(gpy.attn_out_intermediates, gmk.attn_out_intermediates)
        lse_adiff, lse_rdiff = tensor_stats(gpy.attn_lse_intermediates, gmk.attn_lse_intermediates)

        case_name = "forced_non_skip" if force_partitions is not None else "normal_skip"
        return CaseResult(
            case_name=case_name,
            skip_attn_reduction=bool(gmk.skip_attn_reduction),
            max_attn_out_adiff=out_adiff,
            mean_attn_out_rdiff=out_rdiff,
            max_intermediate_adiff=inter_adiff,
            mean_intermediate_rdiff=inter_rdiff,
            max_lse_adiff=lse_adiff,
            mean_lse_rdiff=lse_rdiff,
        )
    finally:
        sched_mod.pick_num_attention_partitions = orig_pick


def print_case(r: CaseResult):
    print("\n===", r.case_name, "===")
    print("skip_attn_reduction:", r.skip_attn_reduction)
    print(
        "attn_out:"
        f" max_adiff={r.max_attn_out_adiff:.6f}"
        f" mean_rdiff={r.mean_attn_out_rdiff:.6f}"
    )
    print(
        "attn_out_intermediates:"
        f" max_adiff={r.max_intermediate_adiff:.6f}"
        f" mean_rdiff={r.mean_intermediate_rdiff:.6f}"
    )
    print(
        "attn_lse_intermediates:"
        f" max_adiff={r.max_lse_adiff:.6f}"
        f" mean_rdiff={r.mean_lse_rdiff:.6f}"
    )


def print_forced_non_skip_details():
    model = base.FakeModel()
    mk_func = importlib.import_module(base.MK_MODULE_NAME).mk_mixtral

    orig_pick = sched_mod.pick_num_attention_partitions
    sched_mod.pick_num_attention_partitions = lambda *args, **kwargs: 2
    try:
        torch.manual_seed(42)
        seed_h = torch.zeros(base.HIDDEN_DIM, device=base.DEVICE, dtype=base.DTYPE)
        normal_(seed_h)

        seed_k = torch.zeros_like(model.stacked_kv_cache[0])
        seed_v = torch.zeros_like(model.stacked_kv_cache[1])
        normal_(seed_k[:, :, : base.SEQ_LEN])
        normal_(seed_v[:, :, : base.SEQ_LEN])

        model.stacked_kv_cache[0].copy_(seed_k)
        model.stacked_kv_cache[1].copy_(seed_v)
        gpy = make_globals(model, seq_len=base.SEQ_LEN)
        gpy.pos_id = base.SEQ_LEN - 1
        gpy.hidden_states.copy_(seed_h)

        nodes_py, end_py = make_dag(gpy, stop_after_op="partial", layer_limit=1)
        sched_py = Schedule(gpy, nodes_py, end_py)
        assigned = assign_to_sms("rr", schedule=sched_py)
        tensorize_instructions(gpy, assigned)
        gpy.barriers.fill_(0)
        PyVM_Interpreter(INSTRUCTION_TO_SOLVER).interpret(gpy, sched_py.get_linear_instructions())

        model.stacked_kv_cache[0].copy_(seed_k)
        model.stacked_kv_cache[1].copy_(seed_v)
        gmk = make_globals(model, seq_len=base.SEQ_LEN)
        gmk.pos_id = base.SEQ_LEN - 1
        gmk.hidden_states.copy_(seed_h)

        nodes_mk, end_mk = make_dag(gmk, stop_after_op="partial", layer_limit=1)
        sched_mk = Schedule(gmk, nodes_mk, end_mk)
        tensorize_instructions(gmk, assigned)
        gmk.barriers.fill_(0)
        interpret_with_mk(gmk, mk_func)
        torch.cuda.synchronize()

        print("\n=== forced_non_skip_details ===")
        py_o = gpy.attn_out_intermediates[:, 0, :].float()
        mk_o = gmk.attn_out_intermediates[:, 0, :].float()
        for h in range(base.NUM_ATTN_HEADS):
            adiff = (py_o[h] - mk_o[h]).abs()
            rdiff = 2 * adiff / (py_o[h].abs() + mk_o[h].abs() + 1e-6)
            print(
                f"head {h}: O_active max_adiff={float(adiff.max()):.6f} "
                f"mean_rdiff={float(rdiff.mean()):.6f}"
            )

        py_l = gpy.attn_lse_intermediates[:, 0].float()
        mk_l = gmk.attn_lse_intermediates[:, 0].float()
        finite = torch.isfinite(py_l) & torch.isfinite(mk_l)
        if finite.any():
            adiff = (py_l[finite] - mk_l[finite]).abs()
            rdiff = 2 * adiff / (py_l[finite].abs() + mk_l[finite].abs() + 1e-6)
            print(
                f"LSE_active_finite: max_adiff={float(adiff.max()):.6f} "
                f"mean_rdiff={float(rdiff.mean()):.6f} count={int(finite.sum())}"
            )
        else:
            print("LSE_active_finite: no finite overlap to compare")
    finally:
        sched_mod.pick_num_attention_partitions = orig_pick


def main():
    normal = build_and_run(force_partitions=None)
    forced = build_and_run(force_partitions=2)

    print_case(normal)
    print_case(forced)

    # Simple classifier
    threshold = 5e-2
    forced_inter_bad = (
        forced.max_intermediate_adiff > threshold
        or forced.mean_intermediate_rdiff > threshold
        or forced.max_lse_adiff > threshold
        or forced.mean_lse_rdiff > threshold
    )

    print("\n=== classification ===")
    if forced_inter_bad:
        print("Divergence appears before skip final-store path.")
        print("Most likely region: partial compute/shared staging (Q/K/V math or store_gqa_rows packing).")
    else:
        print("Forced non-skip intermediates are clean.")
        print("Most likely region: skip-only final attn_out materialization path.")

    print_forced_non_skip_details()


if __name__ == "__main__":
    main()
