"""
Focused diagnostics for the current Mixtral MK router-stage divergence.

This script does not modify kernels. It gathers evidence for where the router
path diverges by comparing:
1) Pre-router state parity at stop_after_op='oproj'
2) Router outputs at stop_after_op='router'
3) MK router outputs against reference math computed from MK pre-router inputs
4) Router instruction encoding/tensorization sanity

Run:
  cd /home/of222/epic-kernel
  uv run python demos/mixtral/debug_router_issue.py
"""

from __future__ import annotations

import importlib

import torch
from torch.nn.init import normal_

import demos.mixtral.debug_mixtral_mk as base
from megakernels.demos.latency.python_vm import rms_norm
from megakernels.mixtral.mk import interpret_with_mk
from megakernels.mixtral.python_vm import INSTRUCTION_TO_SOLVER
from megakernels.mixtral.scheduler import make_dag, make_globals
from megakernels.python_vm import PyVM_Interpreter
from megakernels.scheduler import Schedule, assign_to_sms, tensorize_instructions


def stats(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, float]:
    af = a.float()
    bf = b.float()
    ad = (af - bf).abs()
    rd = 2 * ad / (af.abs() + bf.abs() + 1e-6)
    return float(ad.max().item()), float(ad.mean().item()), float(rd.mean().item())


def build_seed_state(model: base.FakeModel):
    torch.manual_seed(base.DEFAULT_SEED)

    seed_h = torch.zeros(base.HIDDEN_DIM, device=base.DEVICE, dtype=base.DTYPE)
    normal_(seed_h)

    seed_k = torch.zeros_like(model.stacked_kv_cache[0])
    seed_v = torch.zeros_like(model.stacked_kv_cache[1])
    normal_(seed_k[:, :, : base.SEQ_LEN])
    normal_(seed_v[:, :, : base.SEQ_LEN])
    return seed_h, seed_k, seed_v


def run_pair(stop_after_op: str, model: base.FakeModel, mk_func, seed_h, seed_k, seed_v):
    # PyVM globals
    model.stacked_kv_cache[0].copy_(seed_k)
    model.stacked_kv_cache[1].copy_(seed_v)
    gpy = make_globals(model, seq_len=base.SEQ_LEN)
    gpy.pos_id = base.SEQ_LEN - 1
    gpy.hidden_states.copy_(seed_h)
    # Prevent aliasing with model-owned caches
    gpy.k_cache = seed_k.clone()
    gpy.v_cache = seed_v.clone()

    # MK globals
    model.stacked_kv_cache[0].copy_(seed_k)
    model.stacked_kv_cache[1].copy_(seed_v)
    gmk = make_globals(model, seq_len=base.SEQ_LEN)
    gmk.pos_id = base.SEQ_LEN - 1
    gmk.hidden_states.copy_(seed_h)

    nodes_py, end_py = make_dag(gpy, stop_after_op=stop_after_op, layer_limit=base.NUM_LAYERS)
    sched_py = Schedule(gpy, nodes_py, end_py)
    assigned = assign_to_sms("rr", schedule=sched_py)
    tensorize_instructions(gpy, assigned)
    gpy.barriers.fill_(0)

    nodes_mk, end_mk = make_dag(gmk, stop_after_op=stop_after_op, layer_limit=base.NUM_LAYERS)
    sched_mk = Schedule(gmk, nodes_mk, end_mk)
    tensorize_instructions(gmk, assigned)
    gmk.barriers.fill_(0)

    PyVM_Interpreter(INSTRUCTION_TO_SOLVER).interpret(gpy, sched_py.get_linear_instructions())
    interpret_with_mk(gmk, mk_func)
    torch.cuda.synchronize()

    return gpy, gmk, sched_py, assigned


def print_router_instruction_sanity(gmk, assigned):
    # Host-side instruction objects assigned to each SM
    host_router = []
    for sm_idx, queue in enumerate(assigned):
        for slot_idx, ins in enumerate(queue):
            if ins.opcode() == 5:
                host_router.append((sm_idx, slot_idx, ins.serialize()))

    # Tensorized instruction words on device
    instr = gmk.instructions.detach().cpu()
    where = (instr[:, :, 0] == 5).nonzero(as_tuple=False)

    print("\n=== Router instruction sanity ===")
    print(f"host_router_instr_count={len(host_router)}")
    for sm_idx, slot_idx, words in host_router[:8]:
        print(f"  host sm={sm_idx:>3d} slot={slot_idx:>3d} words={words[:6]}")

    print(f"tensor_router_instr_count={where.shape[0]}")
    for i in range(min(8, where.shape[0])):
        sm_idx, slot_idx = where[i].tolist()
        words = instr[sm_idx, slot_idx, :6].tolist()
        print(f"  tensor sm={sm_idx:>3d} slot={slot_idx:>3d} words={words}")


def main() -> None:
    print(f"Loading MK module '{base.MK_MODULE_NAME}'...")
    mk_func = importlib.import_module(base.MK_MODULE_NAME).mk_mixtral

    torch.manual_seed(base.DEFAULT_SEED)
    model = base.FakeModel()
    seed_h, seed_k, seed_v = build_seed_state(model)

    # 1) Pre-router parity at OProj completion
    gpy_oproj, gmk_oproj, _, _ = run_pair("oproj", model, mk_func, seed_h, seed_k, seed_v)

    print("\n=== Pre-router parity (stop_after_op='oproj') ===")
    mad, mean_ad, mean_rd = stats(gpy_oproj.hidden_states, gmk_oproj.hidden_states)
    print(
        "hidden_states py-vs-mk:"
        f" max_adiff={mad:.6f} mean_adiff={mean_ad:.6f} mean_rdiff={mean_rd:.6f}"
    )

    # 2) Router stage outputs
    gpy_router, gmk_router, _, assigned = run_pair("router", model, mk_func, seed_h, seed_k, seed_v)

    print("\n=== Router outputs (stop_after_op='router') ===")
    print(
        "selected_expert_indices:"
        f" py={gpy_router.selected_expert_indices.tolist()}"
        f" mk={gmk_router.selected_expert_indices.tolist()}"
    )
    print(
        "selected_expert_scores:"
        f" py={[round(float(v), 6) for v in gpy_router.selected_expert_scores.float().tolist()]}"
        f" mk={[round(float(v), 6) for v in gmk_router.selected_expert_scores.float().tolist()]}"
    )
    mad, mean_ad, mean_rd = stats(gpy_router.router_normed_hidden, gmk_router.router_normed_hidden)
    print(
        "router_normed_hidden py-vs-mk:"
        f" max_adiff={mad:.6f} mean_adiff={mean_ad:.6f} mean_rdiff={mean_rd:.6f}"
    )

    # 3) Compare MK router outputs against reference computed from MK pre-router inputs
    # Router should consume hidden_states + ffn_ln_weights from the same pre-router state.
    mk_pre_hidden = gmk_oproj.hidden_states.clone()
    mk_ffn_w = gmk_oproj.ffn_ln_weights[0].clone()

    print("\n=== MK pre-router input sanity ===")
    print(
        "pre_hidden stats:"
        f" absmax={float(mk_pre_hidden.float().abs().max()):.6f}"
        f" l2={float(torch.linalg.vector_norm(mk_pre_hidden.float())):.6f}"
    )
    print(
        "ffn_ln_weight stats:"
        f" absmax={float(mk_ffn_w.float().abs().max()):.6f}"
        f" l2={float(torch.linalg.vector_norm(mk_ffn_w.float())):.6f}"
    )

    ref_normed = rms_norm(mk_pre_hidden, mk_ffn_w, gmk_oproj.rms_norm_eps)
    mad, mean_ad, mean_rd = stats(ref_normed, gmk_router.router_normed_hidden)
    print(
        "mk_router_normed vs ref_norm(hidden,ffn_w):"
        f" max_adiff={mad:.6f} mean_adiff={mean_ad:.6f} mean_rdiff={mean_rd:.6f}"
    )

    ref_logits = gmk_oproj.router_weights[0].float() @ ref_normed.float()
    ref_probs = torch.softmax(ref_logits, dim=-1)
    ref_top_scores, ref_top_indices = torch.topk(ref_probs, gmk_oproj.num_experts_per_tok)
    ref_top_scores = ref_top_scores / ref_top_scores.sum()

    print(
        "ref_topk_from_mk_inputs:"
        f" idx={ref_top_indices.tolist()}"
        f" scores={[round(float(v), 6) for v in ref_top_scores.tolist()]}"
    )
    print(
        "mk_topk:"
        f" idx={gmk_router.selected_expert_indices.tolist()}"
        f" scores={[round(float(v), 6) for v in gmk_router.selected_expert_scores.float().tolist()]}"
    )

    # 4) Ensure instruction encoding itself looks sane
    print_router_instruction_sanity(gmk_router, assigned)


if __name__ == "__main__":
    main()
