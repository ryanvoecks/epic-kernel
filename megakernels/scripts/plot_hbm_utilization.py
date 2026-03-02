"""
Plot HBM bandwidth utilization over time from megakernel timing captures.

For each SM and instruction, computes bytes loaded from HBM during the
loader's active window, then aggregates across all SMs into time bins
to show total HBM bandwidth and utilization vs peak.

Usage:
    uv run megakernels/scripts/plot_hbm_utilization.py \
        --input timing_a.pkl --label "round-robin" \
        --peak-bw 8.0 --output hbm_util.html

    # Side-by-side comparison:
    uv run megakernels/scripts/plot_hbm_utilization.py \
        --input timing_a.pkl --label "round-robin" \
        --input2 timing_b.pkl --label2 "DAG scheduler" \
        --peak-bw 8.0 --output hbm_compare.html
"""

import argparse
import pickle
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
from bokeh.plotting import figure, save
from bokeh.resources import INLINE

# ── Constants ────────────────────────────────────────────────────────────────

CYCLE_FREQ_MHZ = 1800.0  # H100/B200 SM clock ~1.8 GHz
CYCLE_FREQ_HZ = CYCLE_FREQ_MHZ * 1e6
BYTES_PER_BF16 = 2
BYTES_PER_F32 = 4

# Llama 1B dimensions
HIDDEN_DIM = 2048
INTERMEDIATE_DIM = 8192
HEAD_DIM = 64
NUM_ATTENTION_HEADS = 32
NUM_KV_HEADS = 8
GQA_RATIO = NUM_ATTENTION_HEADS // NUM_KV_HEADS  # 4
MATVEC_BLOCK_SIZE = 16
KV_BLOCK_SIZE = 16
LM_HEAD_BLOCK_SIZE = 32
VOCAB_SIZE = 128256

# Timing event indices
TEVENT_CONTROLLER_START = 0
TEVENT_CONTROLLER_END = 4
TEVENT_LOADER_START = 5
TEVENT_LOADER_END = 6
TEVENT_LAUNCHER_START = 7
TEVENT_LAUNCHER_END = 8
TEVENT_CONSUMER_START = 11

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

from bokeh.palettes import Category10

_palette = Category10[max(INSTRUCTION_MAP.keys())]
COLOR_MAP = {
    0: "#808080",
    **{k: _palette[k - 1] for k in range(1, max(INSTRUCTION_MAP.keys()) + 1)},
}

# ── Bytes loaded per instruction ─────────────────────────────────────────────


def bytes_loaded_for_instruction(instr, pos_id: int) -> tuple[int, str]:
    """
    Compute bytes loaded from HBM and which timing window to use.

    Returns (bytes, timing_source) where timing_source is:
      - "loader" for ops that load via matvec_pipeline (timing slots 5-6)
      - "launcher" for ops where the launcher does TMA loads (timing slots 7-8)
      - "none" for ops with negligible HBM traffic

    Verified against CUDA source:
      - matvec_pipeline::loader_loop loads st_bf<16,512> tiles (16,384 bytes each)
        4 tiles per iteration = 65,536 bytes per iter.
      - The TMA expect_bytes call confirms: sizeof(bf16) * 2048 * 16 = 65,536.
      - rms_matvec_pipeline additionally TMA-loads norm weights: sv_bf<2048> = 4,096 bytes.
      - Consumer warps load activations directly from HBM (not via loader TMA).
        These are small (4,096 bytes) and happen in the consumer window, not the loader.
        We include them in the total for completeness but they don't dominate.
    """
    from megakernels.demos.latency.instructions import (
        AttentionReduction,
        DownProjResidual,
        LayerNorm_QKV_MatVecRopeAppend,
        LayerNormDoubleMatVecSiLU,
        O_ProjResidual,
        PartialAttention,
        RMS_LM_Head,
    )

    # Bytes per matvec_pipeline iteration (4 × st_bf<16,512> TMA loads)
    BYTES_PER_MATVEC_ITER = 4 * MATVEC_BLOCK_SIZE * 512 * BYTES_PER_BF16  # 65,536

    # Norm weights loaded via TMA in rms_matvec_pipeline: sv_bf<hidden_dim>
    NORM_BYTES = HIDDEN_DIM * BYTES_PER_BF16  # 4,096

    # Consumer direct activation load: 16 warps × sv_bf<128> each = hidden_dim bf16
    CONSUMER_ACT_BYTES = HIDDEN_DIM * BYTES_PER_BF16  # 4,096

    if isinstance(instr, LayerNorm_QKV_MatVecRopeAppend):
        # Loader: matvec_pipeline iterations over QKV weight rows
        # iters = end_block_idx - start_block_idx (parsed_instruction line 41)
        iters = instr.end_output_block_idx - instr.start_output_block_idx
        weight_bytes = iters * BYTES_PER_MATVEC_ITER
        # rms_matvec_pipeline: TMA loads norm weights (once)
        norm_bytes = NORM_BYTES
        # Loader also TMA-loads RoPE cos/sin: 2 × sv_fl<64> = 512 bytes
        rope_bytes = HEAD_DIM * BYTES_PER_F32 * 2  # 512
        # Consumer directly loads hidden_states from HBM
        return weight_bytes + norm_bytes + rope_bytes + CONSUMER_ACT_BYTES, "loader"

    elif isinstance(instr, PartialAttention):
        # LOADER does NOT load KV cache — it only manages pages.
        # LAUNCHER loads K and V cache blocks via TMA (timing slots 7-8).
        # Each KV block: st_bf<kv_block_size, head_dim> = 16×64×2 = 2,048 bytes
        seq_len = pos_id + 1
        total_attn_blocks = (seq_len + KV_BLOCK_SIZE - 1) // KV_BLOCK_SIZE
        blocks_per_partial = (total_attn_blocks + instr.num_partials - 1) // instr.num_partials
        start_blk = instr.partial_idx * blocks_per_partial
        end_blk = min(start_blk + blocks_per_partial, total_attn_blocks)
        num_kv_blocks = max(0, end_blk - start_blk)
        # K + V per block
        kv_bytes = num_kv_blocks * 2 * KV_BLOCK_SIZE * HEAD_DIM * BYTES_PER_BF16
        # Consumer loads Q via cp.async: 4 heads × 64 dims × 2 bytes = 512 bytes
        q_bytes = GQA_RATIO * HEAD_DIM * BYTES_PER_BF16  # 512
        return kv_bytes + q_bytes, "launcher"

    elif isinstance(instr, AttentionReduction):
        # LOADER does NOT load data — only manages pages.
        # LAUNCHER loads LSE intermediates and O intermediates via TMA.
        num_partials = len(instr.reduction_list)
        # LSE: 4 heads × sv_fl<rounded_max_partials>
        # Only num_partials values are meaningful but full vector is loaded
        # sv_fl<160> on B200 (ceil(148/16)*16 = 160) = 640 bytes per head
        rounded_sm_count = ((148 + 15) // 16) * 16  # 160
        lse_bytes = GQA_RATIO * rounded_sm_count * BYTES_PER_F32
        # O intermediates: 4 heads × num_partials × sv_fl<64> = 256 bytes each
        o_bytes = GQA_RATIO * num_partials * HEAD_DIM * BYTES_PER_F32
        return lse_bytes + o_bytes, "launcher"

    elif isinstance(instr, O_ProjResidual):
        # Loader: matvec_pipeline iterations over O_proj weight rows
        # iters = end_block_idx - start_block_idx
        iters = instr.end_block_idx - instr.start_block_idx
        weight_bytes = iters * BYTES_PER_MATVEC_ITER
        # Consumer directly loads attn_out activation from HBM
        return weight_bytes + CONSUMER_ACT_BYTES, "loader"

    elif isinstance(instr, LayerNormDoubleMatVecSiLU):
        # Loader: rms_matvec_pipeline
        # iters = 2 * num_blocks (even=up_weights, odd=gate_weights)
        # (parsed_instruction line 24: iters = 2 * instruction[2])
        iters = 2 * len(instr.block_idxs)
        weight_bytes = iters * BYTES_PER_MATVEC_ITER
        # rms_matvec_pipeline: TMA loads norm weights
        norm_bytes = NORM_BYTES
        # Consumer directly loads hidden_states from HBM
        return weight_bytes + norm_bytes + CONSUMER_ACT_BYTES, "loader"

    elif isinstance(instr, DownProjResidual):
        # Loader: matvec_pipeline iterations over down_proj weight rows
        # Each SM handles a 2048-column slice of the 8192-wide weight
        # (indexed by reduction_block_idx). So per iter it's still 65,536 bytes.
        # iters = end_block_idx - start_block_idx
        iters = instr.end_block_idx - instr.start_block_idx
        weight_bytes = iters * BYTES_PER_MATVEC_ITER
        # Consumer directly loads silu_out activation from HBM (2048-wide slice)
        return weight_bytes + CONSUMER_ACT_BYTES, "loader"

    elif isinstance(instr, RMS_LM_Head):
        # Loader: rms_matvec_pipeline iterations over LM head weight rows
        # Block indices are in units of matvec_block_size=16 rows
        # (NOT lm_head_block_size=32 — the Python scheduler uses 32 but
        #  serializes as pairs of 16-row CUDA blocks)
        iters = instr.end_output_block_idx - instr.start_output_block_idx
        weight_bytes = iters * BYTES_PER_MATVEC_ITER
        # rms_matvec_pipeline: TMA loads norm weights
        norm_bytes = NORM_BYTES
        # Consumer directly loads hidden_states from HBM
        return weight_bytes + norm_bytes + CONSUMER_ACT_BYTES, "loader"

    return 0, "none"


# ── Build utilization data ───────────────────────────────────────────────────


def build_hbm_data(pkl_path: Path, peak_bw_tb_s: float, num_bins: int = 2000):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    timings = data["timings"].cpu().float()  # [num_sms, max_instr, 128]
    instructions_tensor = data["instructions"].cpu()  # [num_sms, max_instr, 32]
    python_instructions = data["python_instructions"]  # list[list[Instruction]]

    num_sms, num_instrs, _ = timings.shape
    peak_bw_bytes_per_cycle = (peak_bw_tb_s * 1e12) / CYCLE_FREQ_HZ

    # Extract pos_id from instructions — find a PartialAttention to get context
    pos_id = 0
    for sm_queue in python_instructions:
        for instr in sm_queue:
            from megakernels.demos.latency.instructions import PartialAttention

            if isinstance(instr, PartialAttention):
                # pos_id is stored in globals, not instruction; estimate from data
                break

    # Try to get pos_id from the pickle data if available
    if "pos_id" in data:
        pos_id = data["pos_id"]
    else:
        # Estimate: use a reasonable default
        pos_id = 41  # prompt_len(32) + ntok - 1

    # Collect per-instruction load events: (start_cycle, end_cycle, bytes, opcode)
    events = []
    per_instr_data = []  # for per-instruction scatter plot

    for sm in range(num_sms):
        for i in range(min(len(python_instructions[sm]), num_instrs)):
            instr = python_instructions[sm][i]

            opcode = instructions_tensor[sm, i, 0].item()
            if opcode == 0:
                continue

            nbytes, timing_source = bytes_loaded_for_instruction(instr, pos_id)
            if nbytes <= 0 or timing_source == "none":
                continue

            # Use the correct timing window based on which warp does the loading
            if timing_source == "loader":
                load_start = timings[sm, i, TEVENT_LOADER_START].item()
                load_end = timings[sm, i, TEVENT_LOADER_END].item()
            elif timing_source == "launcher":
                load_start = timings[sm, i, TEVENT_LAUNCHER_START].item()
                load_end = timings[sm, i, TEVENT_LAUNCHER_END].item()
            else:
                continue

            if load_start <= 0 or load_end <= load_start:
                continue

            events.append((load_start, load_end, nbytes, opcode, sm, i))

            # Per-instruction bandwidth
            duration_cycles = load_end - load_start
            duration_s = duration_cycles / CYCLE_FREQ_HZ
            bw_tb_s = (nbytes / 1e12) / duration_s if duration_s > 0 else 0
            util_pct = (bw_tb_s / peak_bw_tb_s) * 100

            per_instr_data.append(
                {
                    "mid_time_us": (load_start + load_end)
                    / 2
                    / CYCLE_FREQ_MHZ,
                    "duration_us": duration_cycles / CYCLE_FREQ_MHZ,
                    "bytes": nbytes,
                    "bw_tb_s": bw_tb_s,
                    "util_pct": util_pct,
                    "opcode": opcode,
                    "name": INSTRUCTION_MAP.get(opcode, f"op{opcode}"),
                    "color": COLOR_MAP.get(opcode, "#808080"),
                    "sm": sm,
                    "instr_idx": i,
                }
            )

    if not events:
        print(f"  WARNING: No valid load events found in {pkl_path}")
        return None

    # Find time range
    all_starts = [e[0] for e in events]
    all_ends = [e[1] for e in events]
    t_min = min(all_starts)
    t_max = max(all_ends)

    # Create time bins and accumulate bytes per bin
    bin_edges = torch.linspace(t_min, t_max, num_bins + 1)
    bin_width_cycles = (t_max - t_min) / num_bins
    bin_bytes = torch.zeros(num_bins)  # total bytes attributed to each bin
    bin_bytes_by_op = {op: torch.zeros(num_bins) for op in range(8)}

    for start, end, nbytes, opcode, sm, idx in events:
        # Distribute bytes uniformly across bins that overlap this load
        first_bin = max(0, int((start - t_min) / bin_width_cycles))
        last_bin = min(num_bins - 1, int((end - t_min) / bin_width_cycles))

        if first_bin == last_bin:
            bin_bytes[first_bin] += nbytes
            bin_bytes_by_op[opcode][first_bin] += nbytes
        else:
            # Distribute proportionally
            total_duration = end - start
            for b in range(first_bin, last_bin + 1):
                bin_start = t_min + b * bin_width_cycles
                bin_end = bin_start + bin_width_cycles
                overlap_start = max(start, bin_start)
                overlap_end = min(end, bin_end)
                frac = (overlap_end - overlap_start) / total_duration
                bin_bytes[b] += nbytes * frac
                bin_bytes_by_op[opcode][b] += nbytes * frac

    # Convert to bandwidth and utilization
    bin_duration_s = bin_width_cycles / CYCLE_FREQ_HZ
    bin_centers_us = (
        (bin_edges[:-1] + bin_width_cycles / 2 - t_min) / CYCLE_FREQ_MHZ
    ).tolist()

    bw_tb_s = (bin_bytes / 1e12 / bin_duration_s).tolist()
    util_pct = [(b / peak_bw_tb_s) * 100 for b in bw_tb_s]

    bw_by_op = {}
    for op in range(8):
        bw_by_op[op] = (bin_bytes_by_op[op] / 1e12 / bin_duration_s).tolist()

    return {
        "bin_centers_us": bin_centers_us,
        "bw_tb_s": bw_tb_s,
        "util_pct": util_pct,
        "bw_by_op": bw_by_op,
        "per_instr": per_instr_data,
        "peak_bw_tb_s": peak_bw_tb_s,
        "num_sms": num_sms,
        "num_events": len(events),
    }


# ── Figure builders ──────────────────────────────────────────────────────────


def make_bw_figure(hbm_data: dict, title: str, x_range=None):
    bins = hbm_data["bin_centers_us"]
    peak = hbm_data["peak_bw_tb_s"]

    kwargs = dict(
        width=1200,
        height=400,
        title=f"{title} — Aggregate HBM Bandwidth",
        x_axis_label="Time (us)",
        y_axis_label="Bandwidth (TB/s)",
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    if x_range is not None:
        kwargs["x_range"] = x_range
    p = figure(**kwargs)

    # Stacked area by opcode
    bottom = [0.0] * len(bins)
    legend_items = []
    for opcode in [1, 2, 3, 4, 5, 6, 7]:
        top = [b + v for b, v in zip(bottom, hbm_data["bw_by_op"][opcode])]
        src = ColumnDataSource(
            {"x": bins, "top": top, "bottom": bottom}
        )
        r = p.varea(
            x="x",
            y1="bottom",
            y2="top",
            source=src,
            fill_alpha=0.7,
            fill_color=COLOR_MAP[opcode],
        )
        if any(v > 0 for v in hbm_data["bw_by_op"][opcode]):
            legend_items.append(
                LegendItem(label=INSTRUCTION_MAP[opcode], renderers=[r])
            )
        bottom = top

    # Peak bandwidth line
    peak_line = Span(
        location=peak,
        dimension="width",
        line_color="red",
        line_dash="dashed",
        line_width=2,
    )
    p.add_layout(peak_line)

    # Total bandwidth line overlay
    src_total = ColumnDataSource(
        {"x": bins, "y": hbm_data["bw_tb_s"]}
    )
    total_line = p.line(
        "x", "y", source=src_total, line_color="white", line_width=1.5, alpha=0.8
    )
    legend_items.append(LegendItem(label="Total", renderers=[total_line]))

    peak_r = p.line(
        x=[bins[0], bins[-1]],
        y=[peak, peak],
        line_color="red",
        line_dash="dashed",
        line_width=2,
    )
    legend_items.append(
        LegendItem(label=f"Peak ({peak:.1f} TB/s)", renderers=[peak_r])
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


def make_util_figure(hbm_data: dict, title: str, x_range=None):
    bins = hbm_data["bin_centers_us"]

    kwargs = dict(
        width=1200,
        height=300,
        title=f"{title} — HBM Utilization %",
        x_axis_label="Time (us)",
        y_axis_label="Utilization (%)",
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    if x_range is not None:
        kwargs["x_range"] = x_range
    p = figure(**kwargs)

    src = ColumnDataSource({"x": bins, "y": hbm_data["util_pct"]})
    p.line("x", "y", source=src, line_color="dodgerblue", line_width=1.5)
    p.varea(
        x="x",
        y1=0,
        y2="y",
        source=src,
        fill_alpha=0.3,
        fill_color="dodgerblue",
    )

    p.add_layout(
        Span(
            location=100,
            dimension="width",
            line_color="red",
            line_dash="dashed",
            line_width=1,
        )
    )

    p.xaxis.formatter = NumeralTickFormatter(format="0,0")
    p.y_range.start = 0
    p.y_range.end = max(110, max(hbm_data["util_pct"]) * 1.1)

    # Average utilization annotation
    avg_util = sum(hbm_data["util_pct"]) / len(hbm_data["util_pct"])
    p.add_layout(
        Span(
            location=avg_util,
            dimension="width",
            line_color="orange",
            line_dash="dotted",
            line_width=2,
        )
    )

    return p


def make_scatter_figure(hbm_data: dict, title: str, x_range=None):
    per_instr = hbm_data["per_instr"]
    if not per_instr:
        return None

    kwargs = dict(
        width=1200,
        height=350,
        title=f"{title} — Per-Instruction HBM Bandwidth",
        x_axis_label="Time (us)",
        y_axis_label="Bandwidth (TB/s)",
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    if x_range is not None:
        kwargs["x_range"] = x_range
    p = figure(**kwargs)

    # Group by opcode for coloring
    for opcode in [1, 2, 3, 4, 5, 6, 7]:
        subset = [d for d in per_instr if d["opcode"] == opcode]
        if not subset:
            continue

        src = ColumnDataSource(
            {
                "x": [d["mid_time_us"] for d in subset],
                "y": [d["bw_tb_s"] for d in subset],
                "util": [d["util_pct"] for d in subset],
                "duration": [d["duration_us"] for d in subset],
                "bytes": [d["bytes"] for d in subset],
                "sm": [d["sm"] for d in subset],
                "name": [d["name"] for d in subset],
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
                ("SM", "@sm"),
                ("BW", "@y{0.00} TB/s"),
                ("Util", "@util{0.0}%"),
                ("Duration", "@duration{0.00} us"),
                ("Bytes", "@bytes{0,0}"),
            ]
        )
    )

    peak = hbm_data["peak_bw_tb_s"]
    p.add_layout(
        Span(
            location=peak,
            dimension="width",
            line_color="red",
            line_dash="dashed",
            line_width=1,
        )
    )

    p.legend.location = "top_right"
    p.legend.click_policy = "hide"
    p.legend.label_text_font_size = "8pt"
    p.xaxis.formatter = NumeralTickFormatter(format="0,0")
    p.y_range.start = 0

    return p


# ── Entry point ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="HBM utilization from MK timings")
    parser.add_argument("--input", type=Path, required=True, help="Timing pkl file")
    parser.add_argument("--label", type=str, default="Run A")
    parser.add_argument("--input2", type=Path, default=None, help="Second pkl (optional)")
    parser.add_argument("--label2", type=str, default="Run B")
    parser.add_argument(
        "--peak-bw",
        type=float,
        default=8.0,
        help="Peak HBM bandwidth in TB/s (default: 8.0 for B200)",
    )
    parser.add_argument("--bins", type=int, default=2000, help="Number of time bins")
    parser.add_argument(
        "--output", type=Path, default=Path("hbm_util.html"), help="Output HTML file"
    )
    args = parser.parse_args()

    print(f"Processing {args.input} ...")
    data1 = build_hbm_data(args.input, args.peak_bw, args.bins)
    if data1 is None:
        print("ERROR: No data in first input")
        return

    avg_util = sum(data1["util_pct"]) / len(data1["util_pct"])
    print(
        f"  {data1['num_events']} load events across {data1['num_sms']} SMs, "
        f"avg utilization: {avg_util:.1f}%"
    )

    bw_fig = make_bw_figure(data1, args.label)
    util_fig = make_util_figure(data1, args.label, x_range=bw_fig.x_range)
    scatter_fig = make_scatter_figure(data1, args.label, x_range=bw_fig.x_range)

    plots = [bw_fig, util_fig]
    if scatter_fig:
        plots.append(scatter_fig)

    if args.input2:
        print(f"Processing {args.input2} ...")
        data2 = build_hbm_data(args.input2, args.peak_bw, args.bins)
        if data2:
            avg_util2 = sum(data2["util_pct"]) / len(data2["util_pct"])
            print(
                f"  {data2['num_events']} load events across {data2['num_sms']} SMs, "
                f"avg utilization: {avg_util2:.1f}%"
            )

            bw_fig2 = make_bw_figure(data2, args.label2)
            util_fig2 = make_util_figure(
                data2, args.label2, x_range=bw_fig2.x_range
            )
            scatter_fig2 = make_scatter_figure(
                data2, args.label2, x_range=bw_fig2.x_range
            )

            left_col = column(*plots)
            right_plots = [bw_fig2, util_fig2]
            if scatter_fig2:
                right_plots.append(scatter_fig2)
            right_col = column(*right_plots)
            layout = row(left_col, right_col)
        else:
            layout = column(*plots)
    else:
        layout = column(*plots)

    save(
        layout,
        filename=str(args.output),
        title=f"HBM Utilization — {args.label}",
        resources=INLINE,
    )
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
