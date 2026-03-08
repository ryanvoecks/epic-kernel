"""
Plot barrier wait time analysis from megakernel timing captures.

For each SM and instruction, extracts the barrier wait window
(TEVENT_AT_GMEM_WAIT to TEVENT_DONE_GMEM_WAIT, timing slots 44-45)
and produces:

Figures (interactive HTML via Bokeh):
  1. Stall count timeline — how many SMs are simultaneously waiting on
     a barrier at each point in time (stacked area by instruction type).
  2. Barrier wait fraction — fraction of total SM-time spent barrier-waiting.
  3. Per-SM total barrier wait — bar chart showing which SMs wait the most.
  4. Per-instruction scatter — each dot is one instruction's barrier wait,
     x=time, y=wait duration, colored by opcode. Hover shows producer.
  5. Per-layer barrier wait — bar chart (stacked by opcode) showing total
     barrier wait time per transformer layer.
  6. Per-opcode summary — horizontal bar chart of total barrier wait by
     instruction type, showing which operations are the biggest waiters.

Console output:
  - Per-opcode table: count, avg duration, avg wait, wait %, total wait
  - Per-layer table: total wait, instruction count
  - Producer identification: what each stalling opcode is waiting FOR
  - Note on HBM utilization correlation

Usage:
    uv run megakernels/scripts/plot_barrier_waits.py \\
        --input timing_a.pkl --label "round-robin" \\
        --output barrier_waits.html

    # Side-by-side comparison:
    uv run megakernels/scripts/plot_barrier_waits.py \\
        --input timing_a.pkl --label "round-robin" \\
        --input2 timing_b.pkl --label2 "DAG scheduler" \\
        --output barrier_compare.html
"""

import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import torch
from bokeh.layouts import column, row
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    Legend,
    LegendItem,
    NumeralTickFormatter,
    Span,
)
from bokeh.palettes import Category10
from bokeh.plotting import figure, save
from bokeh.resources import INLINE

# ── Constants ────────────────────────────────────────────────────────────────

CYCLE_FREQ_MHZ = 1800.0  # H100/B200 SM clock ~1.8 GHz

# Timing event indices (from include/util.cuh)
TEVENT_CONTROLLER_START = 0
TEVENT_CONTROLLER_END = 4
TEVENT_AT_GMEM_WAIT = 44
TEVENT_DONE_GMEM_WAIT = 45

INSTRUCTION_MAP = {
    0: "No Op",
    1: "RMS QKV MatVec Rope Append",
    2: "Partial Attention",
    3: "Attention Reduction",
    4: "O Proj Residual",
    5: "RMS Double MatVec SiLU",
    6: "Down Proj Residual",
    7: "LM Head",
}

_palette = Category10[max(INSTRUCTION_MAP.keys())]
COLOR_MAP = {
    0: "#808080",
    **{k: _palette[k - 1] for k in range(1, max(INSTRUCTION_MAP.keys()) + 1)},
}

# ── Producer dependency map ──────────────────────────────────────────────────
# Verified against CUDA source (barrier indices in each op's gmem_wait):
#
#   QKV (1): rms_matvec_rope_append.cu:53-54
#       Bar[{layer_idx - 1, OPCODE_DownProjResidual - 1, 0}]
#       → waits on DownProj (6) of the PREVIOUS layer. Layer 0 doesn't wait.
#
#   PartialAttention (2): attention_partial.cu:314-316
#       Bar[{layer_idx, OPCODE_RMS_QKV_MatVecRopeAppend - 1, kv_head + NUM_ATTN_HEADS}]
#       → waits on QKV (1) of the SAME layer.
#
#   AttentionReduction (3): attention_reduction.cu:194-195
#       Bar[{layer_idx, prev_opcode(=2) - 1, kv_head_idx}]
#       → waits on PartialAttention (2) of the SAME layer.
#
#   O_ProjResidual (4): matvec_adds.cu:129, prev_opcode = OPCODE_O_ProjResidual - 1 = 3
#       Bar[{layer, prev_opcode - 1, reduction_block_idx}]
#       → waits on AttentionReduction (3) of the SAME layer.
#
#   UpGate/SiLU (5): upgate.cu:14,36
#       prev_opcode = OPCODE_O_ProjResidual = 4
#       Bar[{layer_idx, prev_opcode - 1, 0}]
#       → waits on O_ProjResidual (4) of the SAME layer.
#
#   DownProj (6): matvec_adds.cu:186, prev_opcode = OPCODE_DownProjResidual - 1 = 5
#       Bar[{layer, prev_opcode - 1, reduction_block_idx}]
#       → waits on UpGate/SiLU (5) of the SAME layer.
#
#   LM Head (7): rms_lm_head.cu:32
#       Bar[{num_layers - 1, OPCODE_DownProjResidual - 1, 0}]
#       → waits on DownProj (6) of the LAST layer.

PRODUCER_MAP = {
    # opcode: (producer_opcode, producer_name, layer_relationship)
    1: (6, "Down Proj Residual", "prev layer"),
    2: (1, "RMS QKV MatVec Rope Append", "same layer"),
    3: (2, "Partial Attention", "same layer"),
    4: (3, "Attention Reduction", "same layer"),
    5: (4, "O Proj Residual", "same layer"),
    6: (5, "RMS Double MatVec SiLU", "same layer"),
    7: (6, "Down Proj Residual", "last layer"),
}


# ── Data extraction ──────────────────────────────────────────────────────────


def build_barrier_data(pkl_path: Path, num_bins: int = 2000):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    timings = data["timings"].cpu().float()  # [num_sms, max_instr, 128]
    instructions = data["instructions"].cpu()  # [num_sms, max_instr, 32]

    num_sms, num_instrs, _ = timings.shape

    # Collect per-instruction barrier wait events
    events = []
    per_instr_data = []
    per_sm_wait_cycles = torch.zeros(num_sms)

    # Per-opcode aggregation
    opcode_stats = defaultdict(lambda: {
        "count": 0,
        "total_wait_us": 0.0,
        "total_instr_us": 0.0,
    })

    # Per-layer aggregation (layer_idx → {opcode → total_wait_us})
    layer_stats = defaultdict(lambda: defaultdict(float))
    layer_instr_count = defaultdict(int)

    for sm in range(num_sms):
        for i in range(num_instrs):
            opcode = instructions[sm, i, 0].item()
            if opcode == 0:
                continue

            # Instruction start/end (controller window)
            instr_start = timings[sm, i, TEVENT_CONTROLLER_START].item()
            instr_end = timings[sm, i, TEVENT_CONTROLLER_END].item()
            if instr_start <= 0 or instr_end <= instr_start:
                continue

            # Extract layer_idx: at instruction word index 1 for opcodes 1-6.
            # Opcode 7 (LM Head) has no layer_idx field; it always operates
            # on the last layer's output, so we label it "lm_head".
            if 1 <= opcode <= 6:
                layer_idx = instructions[sm, i, 1].item()
            else:
                layer_idx = -1  # LM Head

            # Barrier wait window
            wait_start = timings[sm, i, TEVENT_AT_GMEM_WAIT].item()
            wait_end = timings[sm, i, TEVENT_DONE_GMEM_WAIT].item()

            # Skip instructions that don't have barrier waits recorded.
            # wait_start == 0 means the event was never recorded (timing
            # slots are zeroed before each instruction in megakernel.cuh:75-80).
            if wait_start <= 0 or wait_end <= 0:
                continue

            # Sanity: wait_end should be >= wait_start.
            if wait_end < wait_start:
                continue

            wait_duration = wait_end - wait_start
            instr_duration = instr_end - instr_start

            events.append(
                (wait_start, wait_end, instr_start, instr_end, opcode, sm, i)
            )
            per_sm_wait_cycles[sm] += wait_duration

            wait_duration_us = wait_duration / CYCLE_FREQ_MHZ
            instr_duration_us = instr_duration / CYCLE_FREQ_MHZ
            wait_frac = (
                (wait_duration / instr_duration * 100) if instr_duration > 0 else 0
            )

            # Aggregate per opcode
            opcode_stats[opcode]["count"] += 1
            opcode_stats[opcode]["total_wait_us"] += wait_duration_us
            opcode_stats[opcode]["total_instr_us"] += instr_duration_us

            # Aggregate per layer
            layer_stats[layer_idx][opcode] += wait_duration_us
            layer_instr_count[layer_idx] += 1

            # Producer info
            _, producer_name, producer_layer_rel = PRODUCER_MAP.get(
                opcode, (0, "unknown", "")
            )

            per_instr_data.append(
                {
                    "mid_time_us": (wait_start + wait_end) / 2 / CYCLE_FREQ_MHZ,
                    "wait_duration_us": wait_duration_us,
                    "instr_duration_us": instr_duration_us,
                    "wait_frac": wait_frac,
                    "opcode": opcode,
                    "name": INSTRUCTION_MAP.get(opcode, f"op{opcode}"),
                    "color": COLOR_MAP.get(opcode, "#808080"),
                    "sm": sm,
                    "instr_idx": i,
                    "layer_idx": layer_idx,
                    "blocked_by": producer_name,
                    "blocked_by_layer": producer_layer_rel,
                }
            )

    if not events:
        print(f"  WARNING: No barrier wait events found in {pkl_path}")
        return None

    # Find the global time range from ALL instructions (not just barrier events)
    # so that the stall count is relative to the full kernel execution.
    global_start = float("inf")
    global_end = float("-inf")
    total_instr_count = 0
    for sm in range(num_sms):
        for i in range(num_instrs):
            opcode = instructions[sm, i, 0].item()
            if opcode == 0:
                continue
            s = timings[sm, i, TEVENT_CONTROLLER_START].item()
            e = timings[sm, i, TEVENT_CONTROLLER_END].item()
            if s > 0 and e > s:
                global_start = min(global_start, s)
                global_end = max(global_end, e)
                total_instr_count += 1

    if global_start == float("inf"):
        print(f"  WARNING: No valid instructions found in {pkl_path}")
        return None

    # ── Bin barrier wait events into time bins ────────────────────────────
    bin_edges = torch.linspace(global_start, global_end, num_bins + 1)
    bin_width_cycles = (global_end - global_start) / num_bins

    bin_stall_count = torch.zeros(num_bins)
    bin_stall_by_op = {op: torch.zeros(num_bins) for op in range(8)}

    for wait_start, wait_end, _, _, opcode, sm, _ in events:
        wait_duration = wait_end - wait_start
        if wait_duration <= 0:
            continue

        first_bin = max(0, int((wait_start - global_start) / bin_width_cycles))
        last_bin = min(
            num_bins - 1, int((wait_end - global_start) / bin_width_cycles)
        )

        if first_bin == last_bin:
            frac = wait_duration / bin_width_cycles
            bin_stall_count[first_bin] += frac
            bin_stall_by_op[opcode][first_bin] += frac
        else:
            for b in range(first_bin, last_bin + 1):
                bin_start = global_start + b * bin_width_cycles
                bin_end = bin_start + bin_width_cycles
                overlap_start = max(wait_start, bin_start)
                overlap_end = min(wait_end, bin_end)
                overlap = overlap_end - overlap_start
                if overlap > 0:
                    frac = overlap / bin_width_cycles
                    bin_stall_count[b] += frac
                    bin_stall_by_op[opcode][b] += frac

    bin_centers_us = (
        (bin_edges[:-1] + bin_width_cycles / 2 - global_start) / CYCLE_FREQ_MHZ
    ).tolist()

    stall_count = bin_stall_count.tolist()
    stall_frac = [(c / num_sms) * 100 for c in stall_count]

    stall_by_op = {}
    for op in range(8):
        stall_by_op[op] = bin_stall_by_op[op].tolist()

    # ── Per-SM summary ────────────────────────────────────────────────────
    per_sm_wait_us = (per_sm_wait_cycles / CYCLE_FREQ_MHZ).tolist()

    per_sm_total_us = []
    for sm in range(num_sms):
        sm_start = float("inf")
        sm_end = float("-inf")
        for i in range(num_instrs):
            opcode = instructions[sm, i, 0].item()
            if opcode == 0:
                continue
            s = timings[sm, i, TEVENT_CONTROLLER_START].item()
            e = timings[sm, i, TEVENT_CONTROLLER_END].item()
            if s > 0 and e > s:
                sm_start = min(sm_start, s)
                sm_end = max(sm_end, e)
        if sm_start < float("inf"):
            per_sm_total_us.append((sm_end - sm_start) / CYCLE_FREQ_MHZ)
        else:
            per_sm_total_us.append(0.0)

    # Summary statistics
    total_wait_cycles = per_sm_wait_cycles.sum().item()
    total_kernel_cycles = global_end - global_start
    total_sm_cycles = total_kernel_cycles * num_sms
    overall_wait_frac = (
        (total_wait_cycles / total_sm_cycles * 100) if total_sm_cycles > 0 else 0
    )

    # Convert layer_stats to serializable form: {layer_idx: {opcode: us}}
    layer_stats_dict = {
        layer: dict(ops) for layer, ops in sorted(layer_stats.items())
    }
    layer_instr_count_dict = dict(sorted(layer_instr_count.items()))

    return {
        "bin_centers_us": bin_centers_us,
        "stall_count": stall_count,
        "stall_frac": stall_frac,
        "stall_by_op": stall_by_op,
        "per_instr": per_instr_data,
        "per_sm_wait_us": per_sm_wait_us,
        "per_sm_total_us": per_sm_total_us,
        "num_sms": num_sms,
        "num_events": len(events),
        "total_instr_count": total_instr_count,
        "overall_wait_frac": overall_wait_frac,
        "total_wait_us": total_wait_cycles / CYCLE_FREQ_MHZ,
        "total_kernel_us": total_kernel_cycles / CYCLE_FREQ_MHZ,
        "opcode_stats": dict(opcode_stats),
        "layer_stats": layer_stats_dict,
        "layer_instr_count": layer_instr_count_dict,
    }


# ── Console output ───────────────────────────────────────────────────────────


def print_summary(bdata: dict, label: str):
    """Print detailed text summary to console."""
    print(f"\n{'=' * 72}")
    print(f"  {label} — Barrier Wait Summary")
    print(f"{'=' * 72}")

    print(
        f"\n  {bdata['num_events']} barrier wait events across {bdata['num_sms']} SMs "
        f"({bdata['total_instr_count']} total instructions)"
    )
    print(
        f"  Overall: {bdata['overall_wait_frac']:.1f}% of SM-time spent in barrier waits"
    )
    print(
        f"  Kernel duration: {bdata['total_kernel_us']:.1f} us, "
        f"total barrier wait: {bdata['total_wait_us']:.1f} SM-us"
    )

    # ── Per-opcode table ──────────────────────────────────────────────────
    print(f"\n  {'─' * 68}")
    print("  Per-Opcode Barrier Wait Breakdown")
    print(f"  {'─' * 68}")
    print(
        f"  {'Opcode':<30s} {'Count':>6s} {'Avg Wait':>10s} "
        f"{'Avg Dur':>10s} {'Wait%':>7s} {'Tot Wait':>10s}"
    )
    print(f"  {'':-<30s} {'':-<6s} {'':-<10s} {'':-<10s} {'':-<7s} {'':-<10s}")

    opcode_stats = bdata["opcode_stats"]
    # Sort by total wait descending
    sorted_ops = sorted(
        opcode_stats.items(), key=lambda x: x[1]["total_wait_us"], reverse=True
    )
    for opcode, stats in sorted_ops:
        name = INSTRUCTION_MAP.get(opcode, f"op{opcode}")
        count = stats["count"]
        avg_wait = stats["total_wait_us"] / count if count > 0 else 0
        avg_dur = stats["total_instr_us"] / count if count > 0 else 0
        wait_pct = (
            (stats["total_wait_us"] / stats["total_instr_us"] * 100)
            if stats["total_instr_us"] > 0
            else 0
        )
        total_wait = stats["total_wait_us"]
        print(
            f"  {name:<30s} {count:>6d} {avg_wait:>9.2f}u "
            f"{avg_dur:>9.2f}u {wait_pct:>6.1f}% {total_wait:>9.1f}u"
        )

    # ── Producer identification ───────────────────────────────────────────
    print(f"\n  {'─' * 68}")
    print("  Barrier Dependency Chain (what each operation waits FOR)")
    print(f"  {'─' * 68}")
    for opcode in sorted(PRODUCER_MAP.keys()):
        if opcode not in opcode_stats:
            continue
        _, prod_name, prod_rel = PRODUCER_MAP[opcode]
        waiter = INSTRUCTION_MAP[opcode]
        total = opcode_stats[opcode]["total_wait_us"]
        print(
            f"  {waiter:<30s} waits on {prod_name} ({prod_rel})"
            f"  [{total:.1f} us total]"
        )

    # ── Per-layer table ───────────────────────────────────────────────────
    layer_stats = bdata["layer_stats"]
    layer_counts = bdata["layer_instr_count"]
    if layer_stats:
        print(f"\n  {'─' * 68}")
        print("  Per-Layer Barrier Wait")
        print(f"  {'─' * 68}")
        print(f"  {'Layer':<10s} {'# Waits':>8s} {'Total Wait':>12s} {'Top Waiter':<30s}")
        print(f"  {'':-<10s} {'':-<8s} {'':-<12s} {'':-<30s}")

        for layer_idx in sorted(layer_stats.keys()):
            ops = layer_stats[layer_idx]
            total = sum(ops.values())
            count = layer_counts[layer_idx]
            # Find the opcode contributing most wait in this layer
            top_op = max(ops.items(), key=lambda x: x[1])
            top_name = INSTRUCTION_MAP.get(top_op[0], f"op{top_op[0]}")
            top_pct = (top_op[1] / total * 100) if total > 0 else 0

            layer_label = f"Layer {layer_idx}" if layer_idx >= 0 else "LM Head"
            print(
                f"  {layer_label:<10s} {count:>8d} {total:>11.1f}u "
                f"{top_name} ({top_pct:.0f}%)"
            )

    # ── HBM correlation note ──────────────────────────────────────────────
    print(f"\n  {'─' * 68}")
    print("  Note: Time axis (us) uses the same coordinate system as")
    print("  plot_hbm_utilization.py. To correlate barrier stalls with HBM")
    print("  underutilization, generate both plots and compare time ranges")
    print("  where barrier stall count is high with HBM bandwidth dips.")
    print(f"  {'─' * 68}\n")


# ── Figure builders ──────────────────────────────────────────────────────────


def make_stall_count_figure(bdata: dict, title: str, x_range=None):
    """Stacked area: number of SMs simultaneously waiting on barriers."""
    bins = bdata["bin_centers_us"]

    kwargs = dict(
        width=1200,
        height=400,
        title=f"{title} — SMs Stalled on Barriers",
        x_axis_label="Time (us)",
        y_axis_label="# SMs waiting",
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    if x_range is not None:
        kwargs["x_range"] = x_range
    p = figure(**kwargs)

    bottom = [0.0] * len(bins)
    legend_items = []
    for opcode in [1, 2, 3, 4, 5, 6, 7]:
        values = bdata["stall_by_op"][opcode]
        if not any(v > 0 for v in values):
            continue
        top = [b + v for b, v in zip(bottom, values)]
        src = ColumnDataSource({"x": bins, "top": top, "bottom": bottom})
        r = p.varea(
            x="x",
            y1="bottom",
            y2="top",
            source=src,
            fill_alpha=0.7,
            fill_color=COLOR_MAP[opcode],
        )
        legend_items.append(
            LegendItem(label=INSTRUCTION_MAP[opcode], renderers=[r])
        )
        bottom = top

    src_total = ColumnDataSource({"x": bins, "y": bdata["stall_count"]})
    total_line = p.line(
        "x", "y", source=src_total, line_color="white", line_width=1.5, alpha=0.8
    )
    legend_items.append(LegendItem(label="Total", renderers=[total_line]))

    num_sms = bdata["num_sms"]
    sm_line = p.line(
        x=[bins[0], bins[-1]],
        y=[num_sms, num_sms],
        line_color="red",
        line_dash="dashed",
        line_width=1,
    )
    legend_items.append(
        LegendItem(label=f"All SMs ({num_sms})", renderers=[sm_line])
    )

    legend = Legend(
        items=legend_items,
        location="top_right",
        click_policy="hide",
        label_text_font_size="9pt",
    )
    p.add_layout(legend, "right")

    p.xaxis.formatter = NumeralTickFormatter(format="0,0")
    p.y_range.start = 0

    return p


def make_stall_frac_figure(bdata: dict, title: str, x_range=None):
    """Line + area: fraction of SMs that are barrier-waiting over time."""
    bins = bdata["bin_centers_us"]

    kwargs = dict(
        width=1200,
        height=300,
        title=f"{title} — Barrier Wait Fraction (% of SMs)",
        x_axis_label="Time (us)",
        y_axis_label="SMs waiting (%)",
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    if x_range is not None:
        kwargs["x_range"] = x_range
    p = figure(**kwargs)

    src = ColumnDataSource({"x": bins, "y": bdata["stall_frac"]})
    p.line("x", "y", source=src, line_color="dodgerblue", line_width=1.5)
    p.varea(
        x="x",
        y1=0,
        y2="y",
        source=src,
        fill_alpha=0.3,
        fill_color="dodgerblue",
    )

    avg_frac = sum(bdata["stall_frac"]) / len(bdata["stall_frac"])
    p.add_layout(
        Span(
            location=avg_frac,
            dimension="width",
            line_color="orange",
            line_dash="dotted",
            line_width=2,
        )
    )

    p.xaxis.formatter = NumeralTickFormatter(format="0,0")
    p.y_range.start = 0
    p.y_range.end = max(110, max(bdata["stall_frac"]) * 1.1)

    return p


def make_per_sm_figure(bdata: dict, title: str):
    """Bar chart: total barrier wait time per SM."""
    num_sms = bdata["num_sms"]
    wait_us = bdata["per_sm_wait_us"]
    total_us = bdata["per_sm_total_us"]
    wait_frac = [
        (w / t * 100) if t > 0 else 0.0 for w, t in zip(wait_us, total_us)
    ]

    src = ColumnDataSource(
        {
            "sm": list(range(num_sms)),
            "wait_us": wait_us,
            "total_us": total_us,
            "wait_frac": wait_frac,
        }
    )

    p = figure(
        width=1200,
        height=300,
        title=f"{title} — Per-SM Total Barrier Wait",
        x_axis_label="SM index",
        y_axis_label="Barrier wait (us)",
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )

    bars = p.vbar(
        x="sm",
        top="wait_us",
        source=src,
        width=0.8,
        color="dodgerblue",
        alpha=0.7,
    )

    p.add_tools(
        HoverTool(
            renderers=[bars],
            tooltips=[
                ("SM", "@sm"),
                ("Wait", "@wait_us{0.00} us"),
                ("Total", "@total_us{0.00} us"),
                ("Wait %", "@wait_frac{0.0}%"),
            ],
        )
    )

    avg_wait = sum(wait_us) / len(wait_us)
    p.add_layout(
        Span(
            location=avg_wait,
            dimension="width",
            line_color="orange",
            line_dash="dotted",
            line_width=2,
        )
    )

    p.y_range.start = 0

    return p


def make_scatter_figure(bdata: dict, title: str, x_range=None):
    """Scatter: per-instruction barrier wait duration vs time.

    Hover tooltip shows the producer (what this instruction is waiting for).
    """
    per_instr = bdata["per_instr"]
    if not per_instr:
        return None

    kwargs = dict(
        width=1200,
        height=350,
        title=f"{title} — Per-Instruction Barrier Wait Duration",
        x_axis_label="Time (us)",
        y_axis_label="Wait duration (us)",
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    if x_range is not None:
        kwargs["x_range"] = x_range
    p = figure(**kwargs)

    for opcode in [1, 2, 3, 4, 5, 6, 7]:
        subset = [d for d in per_instr if d["opcode"] == opcode]
        if not subset:
            continue

        src = ColumnDataSource(
            {
                "x": [d["mid_time_us"] for d in subset],
                "y": [d["wait_duration_us"] for d in subset],
                "wait_frac": [d["wait_frac"] for d in subset],
                "instr_dur": [d["instr_duration_us"] for d in subset],
                "sm": [d["sm"] for d in subset],
                "name": [d["name"] for d in subset],
                "instr_idx": [d["instr_idx"] for d in subset],
                "layer": [
                    (f"L{d['layer_idx']}" if d["layer_idx"] >= 0 else "LM")
                    for d in subset
                ],
                "blocked_by": [d["blocked_by"] for d in subset],
                "blocked_by_layer": [d["blocked_by_layer"] for d in subset],
            }
        )
        p.scatter(
            "x",
            "y",
            source=src,
            color=COLOR_MAP[opcode],
            size=4,
            alpha=0.5,
            legend_label=INSTRUCTION_MAP[opcode],
        )

    p.add_tools(
        HoverTool(
            tooltips=[
                ("Op", "@name"),
                ("Layer", "@layer"),
                ("SM", "@sm"),
                ("Instr #", "@instr_idx"),
                ("Wait", "@y{0.00} us"),
                ("Instr dur", "@instr_dur{0.00} us"),
                ("Wait %", "@wait_frac{0.0}%"),
                ("Blocked by", "@blocked_by (@blocked_by_layer)"),
            ]
        )
    )

    p.legend.location = "top_right"
    p.legend.click_policy = "hide"
    p.legend.label_text_font_size = "8pt"
    p.xaxis.formatter = NumeralTickFormatter(format="0,0")
    p.y_range.start = 0

    return p


def make_per_layer_figure(bdata: dict, title: str):
    """Stacked bar chart: total barrier wait per layer, broken down by opcode."""
    layer_stats = bdata["layer_stats"]
    if not layer_stats:
        return None

    sorted_layers = sorted(layer_stats.keys())
    layer_labels = [
        f"L{l}" if l >= 0 else "LM" for l in sorted_layers
    ]

    p = figure(
        width=1200,
        height=300,
        title=f"{title} — Per-Layer Total Barrier Wait",
        x_axis_label="Layer",
        y_axis_label="Barrier wait (us)",
        x_range=layer_labels,
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )

    # Build stacked bar data
    legend_items = []
    bottoms = [0.0] * len(sorted_layers)

    for opcode in [1, 2, 3, 4, 5, 6, 7]:
        values = [
            layer_stats[l].get(opcode, 0.0) for l in sorted_layers
        ]
        if not any(v > 0 for v in values):
            continue

        tops = [b + v for b, v in zip(bottoms, values)]

        src = ColumnDataSource(
            {
                "x": layer_labels,
                "top": tops,
                "bottom": bottoms,
                "value": values,
                "name": [INSTRUCTION_MAP[opcode]] * len(sorted_layers),
            }
        )
        r = p.vbar(
            x="x",
            top="top",
            bottom="bottom",
            source=src,
            width=0.7,
            color=COLOR_MAP[opcode],
            alpha=0.8,
        )
        legend_items.append(
            LegendItem(label=INSTRUCTION_MAP[opcode], renderers=[r])
        )
        bottoms = tops

    if legend_items:
        legend = Legend(
            items=legend_items,
            location="top_right",
            click_policy="hide",
            label_text_font_size="8pt",
        )
        p.add_layout(legend, "right")

    p.y_range.start = 0

    return p


def make_per_opcode_figure(bdata: dict, title: str):
    """Horizontal bar chart: total barrier wait by instruction type."""
    opcode_stats = bdata["opcode_stats"]
    if not opcode_stats:
        return None

    # Sort by total wait descending
    sorted_ops = sorted(
        opcode_stats.items(), key=lambda x: x[1]["total_wait_us"], reverse=True
    )

    names = [INSTRUCTION_MAP.get(op, f"op{op}") for op, _ in sorted_ops]
    total_waits = [s["total_wait_us"] for _, s in sorted_ops]
    counts = [s["count"] for _, s in sorted_ops]
    colors = [COLOR_MAP.get(op, "#808080") for op, _ in sorted_ops]
    avg_waits = [
        s["total_wait_us"] / s["count"] if s["count"] > 0 else 0
        for _, s in sorted_ops
    ]
    wait_pcts = [
        (s["total_wait_us"] / s["total_instr_us"] * 100)
        if s["total_instr_us"] > 0
        else 0
        for _, s in sorted_ops
    ]
    # Producer info for hover
    blocked_by = []
    for op, _ in sorted_ops:
        if op in PRODUCER_MAP:
            _, pname, prel = PRODUCER_MAP[op]
            blocked_by.append(f"{pname} ({prel})")
        else:
            blocked_by.append("n/a")

    src = ColumnDataSource(
        {
            "name": names,
            "total_wait": total_waits,
            "count": counts,
            "color": colors,
            "avg_wait": avg_waits,
            "wait_pct": wait_pcts,
            "blocked_by": blocked_by,
        }
    )

    p = figure(
        width=1200,
        height=300,
        title=f"{title} — Total Barrier Wait by Instruction Type",
        x_axis_label="Total barrier wait (us)",
        y_range=list(reversed(names)),
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )

    bars = p.hbar(
        y="name",
        right="total_wait",
        source=src,
        height=0.6,
        color="color",
        alpha=0.8,
    )

    p.add_tools(
        HoverTool(
            renderers=[bars],
            tooltips=[
                ("Op", "@name"),
                ("Total wait", "@total_wait{0.0} us"),
                ("Count", "@count"),
                ("Avg wait", "@avg_wait{0.00} us"),
                ("Wait % of instr", "@wait_pct{0.0}%"),
                ("Blocked by", "@blocked_by"),
            ],
        )
    )

    p.x_range.start = 0

    return p


# ── Entry point ──────────────────────────────────────────────────────────────


def build_plots(bdata: dict, label: str):
    """Build all figures for one dataset. Returns list of figures."""
    stall_fig = make_stall_count_figure(bdata, label)
    frac_fig = make_stall_frac_figure(bdata, label, x_range=stall_fig.x_range)
    sm_fig = make_per_sm_figure(bdata, label)
    scatter_fig = make_scatter_figure(bdata, label, x_range=stall_fig.x_range)
    layer_fig = make_per_layer_figure(bdata, label)
    opcode_fig = make_per_opcode_figure(bdata, label)

    plots = [stall_fig, frac_fig, sm_fig]
    if scatter_fig:
        plots.append(scatter_fig)
    if layer_fig:
        plots.append(layer_fig)
    if opcode_fig:
        plots.append(opcode_fig)

    return plots


def main():
    parser = argparse.ArgumentParser(
        description="Barrier wait analysis from MK timings"
    )
    parser.add_argument("--input", type=Path, required=True, help="Timing pkl file")
    parser.add_argument("--label", type=str, default="Run A")
    parser.add_argument(
        "--input2", type=Path, default=None, help="Second pkl (optional)"
    )
    parser.add_argument("--label2", type=str, default="Run B")
    parser.add_argument("--bins", type=int, default=2000, help="Number of time bins")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("barrier_waits.html"),
        help="Output HTML file",
    )
    args = parser.parse_args()

    print(f"Processing {args.input} ...")
    data1 = build_barrier_data(args.input, args.bins)
    if data1 is None:
        print("ERROR: No data in first input")
        return

    print_summary(data1, args.label)
    plots1 = build_plots(data1, args.label)

    if args.input2:
        print(f"Processing {args.input2} ...")
        data2 = build_barrier_data(args.input2, args.bins)
        if data2:
            print_summary(data2, args.label2)
            plots2 = build_plots(data2, args.label2)

            left_col = column(*plots1)
            right_col = column(*plots2)
            layout = row(left_col, right_col)
        else:
            layout = column(*plots1)
    else:
        layout = column(*plots1)

    save(
        layout,
        filename=str(args.output),
        title=f"Barrier Waits — {args.label}",
        resources=INLINE,
    )
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
