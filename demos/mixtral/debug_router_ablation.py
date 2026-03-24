"""
Controlled router ablation to validate the router-stage failure mechanism.

This sets deterministic, simplified weights so expected router behavior is easy
to reason about while still running through the same MK opcode path.

Run:
  cd /home/of222/epic-kernel
  uv run python demos/mixtral/debug_router_ablation.py
"""

from __future__ import annotations

import importlib

import torch

import demos.mixtral.debug_mixtral_mk as base
from megakernels.mixtral.mk import interpret_with_mk
from megakernels.mixtral.python_vm import INSTRUCTION_TO_SOLVER
from megakernels.mixtral.scheduler import make_dag, make_globals
from megakernels.python_vm import PyVM_Interpreter
from megakernels.scheduler import Schedule, assign_to_sms, tensorize_instructions


def run_pair(model: base.FakeModel, hidden_seed: torch.Tensor, mk_prefill: float | None = None):
    mk_func = importlib.import_module(base.MK_MODULE_NAME).mk_mixtral

    gpy = make_globals(model, seq_len=base.SEQ_LEN)
    gpy.pos_id = base.SEQ_LEN - 1
    gpy.hidden_states.copy_(hidden_seed)
    gpy.k_cache.zero_()
    gpy.v_cache.zero_()

    gmk = make_globals(model, seq_len=base.SEQ_LEN)
    gmk.pos_id = base.SEQ_LEN - 1
    gmk.hidden_states.copy_(hidden_seed)
    gmk.k_cache.zero_()
    gmk.v_cache.zero_()
    if mk_prefill is not None:
        gmk.router_normed_hidden.fill_(mk_prefill)

    nodes_py, end_py = make_dag(gpy, stop_after_op="router", layer_limit=base.NUM_LAYERS)
    sched_py = Schedule(gpy, nodes_py, end_py)
    assigned = assign_to_sms("rr", schedule=sched_py)
    tensorize_instructions(gpy, assigned)
    gpy.barriers.fill_(0)

    nodes_mk, end_mk = make_dag(gmk, stop_after_op="router", layer_limit=base.NUM_LAYERS)
    _ = Schedule(gmk, nodes_mk, end_mk)
    tensorize_instructions(gmk, assigned)
    gmk.barriers.fill_(0)

    PyVM_Interpreter(INSTRUCTION_TO_SOLVER).interpret(gpy, sched_py.get_linear_instructions())
    interpret_with_mk(gmk, mk_func)
    torch.cuda.synchronize()

    return gpy, gmk


def main() -> None:
    torch.manual_seed(base.DEFAULT_SEED)
    model = base.FakeModel()

    # Simplify upstream attention contribution so hidden mostly passes through.
    model.stacked_params.qkv_proj.zero_()
    model.stacked_params.o_proj.zero_()

    # Deterministic router math setup.
    model.stacked_params.ffn_ln_weight.fill_(1.0)
    model.stacked_params.router_weight.zero_()
    model.stacked_params.router_weight[0, 0].fill_(1.0)
    model.stacked_params.router_weight[0, 1].fill_(-1.0)
    model.stacked_params.router_weight[0, 2].fill_(0.25)
    model.stacked_params.router_weight[0, 3].fill_(-0.25)

    hidden_seed = torch.linspace(
        -1.0, 1.0, base.HIDDEN_DIM, device=base.DEVICE, dtype=base.DTYPE
    )

    gpy, gmk = run_pair(model, hidden_seed, mk_prefill=None)

    py_norm = gpy.router_normed_hidden.float()
    mk_norm = gmk.router_normed_hidden.float()
    ad = (py_norm - mk_norm).abs()

    print("=== Controlled router ablation ===")
    print(
        "router_normed_hidden:"
        f" py_absmax={float(py_norm.abs().max()):.6f}"
        f" mk_absmax={float(mk_norm.abs().max()):.6f}"
        f" max_adiff={float(ad.max()):.6f}"
        f" mean_adiff={float(ad.mean()):.6f}"
    )
    print(
        "selected_expert_indices:"
        f" py={gpy.selected_expert_indices.tolist()}"
        f" mk={gmk.selected_expert_indices.tolist()}"
    )
    print(
        "selected_expert_scores:"
        f" py={[round(float(v), 6) for v in gpy.selected_expert_scores.float().tolist()]}"
        f" mk={[round(float(v), 6) for v in gmk.selected_expert_scores.float().tolist()]}"
    )

    # Sentinel test: prefill MK router_normed_hidden before kernel launch.
    # If router output depends on this prefill, router is reading stale global
    # buffer rather than freshly computed normed activations.
    _, gmk_prefill = run_pair(model, hidden_seed, mk_prefill=1.0)
    mk_prefill_norm = gmk_prefill.router_normed_hidden.float()
    print("\n=== Sentinel prefill test (mk_prefill=1.0) ===")
    print(
        "router_normed_hidden after run:"
        f" mk_absmax={float(mk_prefill_norm.abs().max()):.6f}"
        f" first4={[round(float(v), 4) for v in mk_prefill_norm[:4].tolist()]}"
    )
    print(
        "selected_expert_indices:"
        f" mk={gmk_prefill.selected_expert_indices.tolist()}"
    )
    print(
        "selected_expert_scores:"
        f" mk={[round(float(v), 6) for v in gmk_prefill.selected_expert_scores.float().tolist()]}"
    )


if __name__ == "__main__":
    main()
