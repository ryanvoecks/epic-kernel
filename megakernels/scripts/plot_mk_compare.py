"""
Side-by-side SM-level Gantt chart comparing two MegaKernel timing captures,
with full sub-instruction event markers and k-slot lane splitting.

Usage:
    uv run megakernels/scripts/plot_mk_compare.py \\
        --left  timing_128_matched.pkl  --left-label  "128 output tokens" \\
        --right timing_4096_matched.pkl --right-label "4096 output tokens" \\
        --depth 2 --output mk_compare.html
"""

import argparse
import pickle
import time
from pathlib import Path

from bokeh.layouts import row
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    Legend,
    LegendItem,
    NumeralTickFormatter,
)
from bokeh.plotting import figure, save
from bokeh.resources import CDN

# ── Constants ─────────────────────────────────────────────────────────────────

CYCLE_FREQ_MHZ = 1800.0

BAR_HEIGHT_RATIO = 2 / 3
BG_EVEN = "white"
BG_ODD  = "#404040"

# ── Instruction metadata ──────────────────────────────────────────────────────

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

# ── Sub-instruction event markers ─────────────────────────────────────────────
# Indices into the 128-slot timing dimension.
# Convention from the kernel source (TEVENT_* constants):
#   0  controller_start    4  controller_end
#   5  loader_start        6  loader_end
#   7  launcher_start      8  launcher_end
#   9  storer_start       10  storer_end
#  11  consumer_start     12  consumer_end
#  44  at_gmem_wait       45  done_gmem_wait
#  46  at_gmem_store      47  done_gmem_store
#  48  first_load         49  first_use
#  50  first_store        51  last_load
#  52  last_use           53  last_store

PINK = "#FF80FF"
CYAN = "#00FFFF"
START_SYM = "circle"
END_SYM   = "cross"

EVENT_MAPPING = {
    0:  ("controller_start",  START_SYM, "green"),
    4:  ("controller_end",    END_SYM,   "green"),
    5:  ("loader_start",      START_SYM, "orange"),
    6:  ("loader_end",        END_SYM,   "orange"),
    7:  ("launcher_start",    START_SYM, "blue"),
    8:  ("launcher_end",      END_SYM,   "blue"),
    9:  ("storer_start",      START_SYM, "red"),
    10: ("storer_end",        END_SYM,   "red"),
    11: ("consumer_start",    START_SYM, "purple"),
    12: ("consumer_end",      END_SYM,   "purple"),
    44: ("at_gmem_wait",      START_SYM, "yellow"),
    45: ("done_gmem_wait",    END_SYM,   "yellow"),
    46: ("at_gmem_store",     START_SYM, PINK),
    47: ("done_gmem_store",   END_SYM,   PINK),
    48: ("first_load",        START_SYM, CYAN),
    49: ("first_use",         START_SYM, "white"),
    50: ("first_store",       START_SYM, "black"),
    51: ("last_load",         END_SYM,   CYAN),
    52: ("last_use",          END_SYM,   "white"),
    53: ("last_store",        END_SYM,   "black"),
}

MARKER_STYLES = {
    name: {"marker": sym, "color": col, "size": 8}
    for _, (name, sym, col) in EVENT_MAPPING.items()
}

# ── Figure builder ────────────────────────────────────────────────────────────

def make_mk_figure(
    pkl_path: Path,
    title: str,
    k_slots: int = 2,
    show_events: bool = True,
    x_range=None,
) -> figure:
    t0 = time.time()
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    timings      = data["timings"].cpu().float()    # [num_sms, max_instr, 128]
    instructions = data["instructions"].cpu()       # [num_sms, max_instr, 32]
    timings_us   = timings / CYCLE_FREQ_MHZ
    num_sms, num_instrs, _ = timings.shape

    bar_height = BAR_HEIGHT_RATIO / k_slots
    bar_margin = (1.0 - BAR_HEIGHT_RATIO) / (k_slots + 1)

    bar_data = {k: [] for k in
                ("left", "right", "bottom", "top", "color",
                 "name", "proc", "instr_idx", "start_us", "end_us", "duration")}
    marker_data = {name: {"x": [], "y": [], "event": [], "proc": [], "instr_idx": []}
                   for name in MARKER_STYLES}

    t_min = float("inf")
    t_max = float("-inf")

    for sm in range(num_sms):
        for i in range(num_instrs):
            start = timings_us[sm, i, 0].item()
            end   = timings_us[sm, i, 4].item()
            if start <= 0 or end <= start:
                continue

            itype    = instructions[sm, i, 0].item()
            dur      = end - start
            t_min    = min(t_min, start)
            t_max    = max(t_max, end)

            slot   = i % k_slots
            y_base = sm
            bottom = y_base - 0.5 + slot * bar_height + (slot + 1) * bar_margin
            top    = bottom + bar_height
            marker_y = (bottom + top) / 2

            bar_data["left"].append(start)
            bar_data["right"].append(end)
            bar_data["bottom"].append(bottom)
            bar_data["top"].append(top)
            bar_data["color"].append(COLOR_MAP.get(itype, "#808080"))
            bar_data["name"].append(INSTRUCTION_MAP.get(itype, f"op{itype}"))
            bar_data["proc"].append(sm)
            bar_data["instr_idx"].append(i)
            bar_data["start_us"].append(start)
            bar_data["end_us"].append(end)
            bar_data["duration"].append(dur)

            if show_events:
                for ev_id, (ev_name, _, _) in EVENT_MAPPING.items():
                    ev_t = timings_us[sm, i, ev_id].item()
                    if ev_t > 0 and start <= ev_t <= end:
                        marker_data[ev_name]["x"].append(ev_t)
                        marker_data[ev_name]["y"].append(marker_y)
                        marker_data[ev_name]["event"].append(ev_name)
                        marker_data[ev_name]["proc"].append(sm)
                        marker_data[ev_name]["instr_idx"].append(i)

    print(f"  {pkl_path.name}: {num_sms} SMs, data in {time.time()-t0:.1f}s")

    if t_min == float("inf"):
        t_min, t_max = 0.0, 1.0
    pad = 0.05 * max(1.0, t_max - t_min)

    plot_h = max(400, min(3000, int(num_sms * 20 * max(1, k_slots / 1.5))))
    plot_w = max(800, min(6000, int(plot_h * max(5, (t_max - t_min) / (num_sms * 5)))))

    p = figure(
        width=plot_w,
        height=plot_h,
        title=title,
        x_axis_label="Time (μs)",
        y_axis_label="SM index",
        tools="pan,wheel_zoom,box_zoom,reset,save",
        y_range=(-0.5, num_sms - 0.5),
        x_range=x_range if x_range is not None else (t_min - pad, t_max + pad),
    )
    p.xaxis.formatter = NumeralTickFormatter(format="0,0.0")
    p.yaxis.ticker = list(range(0, num_sms, max(1, num_sms // 20)))
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None

    # Alternating SM row backgrounds
    bg_w = (t_max + pad) - (t_min - pad)
    bg_cx = (t_min - pad) + bg_w / 2
    p.rect(
        x=[bg_cx] * num_sms,
        y=list(range(num_sms)),
        width=[bg_w] * num_sms,
        height=[1.0] * num_sms,
        fill_color=[BG_EVEN if s % 2 == 0 else BG_ODD for s in range(num_sms)],
        line_color=None,
        level="underlay",
    )

    # Instruction bars
    bar_src = ColumnDataSource(bar_data)
    bars = p.quad(left="left", right="right", bottom="bottom", top="top",
                  color="color", source=bar_src, name="bars")
    p.add_tools(HoverTool(renderers=[bars], tooltips=[
        ("SM",       "@proc"),
        ("Instr #",  "@instr_idx"),
        ("Type",     "@name"),
        ("Start",    "@start_us{0,0.00} μs"),
        ("End",      "@end_us{0,0.00} μs"),
        ("Duration", "@duration{0,0.00} μs"),
    ]))

    # Sub-instruction event markers
    marker_renderers = {}
    if show_events:
        for ev_name, style in MARKER_STYLES.items():
            md = marker_data[ev_name]
            if not md["x"]:
                continue
            src = ColumnDataSource(md)
            r = p.scatter(x="x", y="y", source=src,
                          marker=style["marker"], color=style["color"],
                          size=style["size"], name=f"ev_{ev_name}")
            marker_renderers[ev_name] = r
        if marker_renderers:
            p.add_tools(HoverTool(
                renderers=list(marker_renderers.values()),
                tooltips=[
                    ("SM",      "@proc"),
                    ("Instr #", "@instr_idx"),
                    ("Event",   "@event"),
                    ("Time",    "@x{0,0.00} μs"),
                ],
            ))

    # Legend — only instruction types actually present
    seen: set = set()
    instr_items = []
    for itype_t, name in zip(bar_data["color"], bar_data["name"]):
        icode = next((k for k, v in COLOR_MAP.items() if v == itype_t), None)
        if icode not in seen:
            seen.add(icode)
            r = p.rect(x=0, y=0, width=0, height=0,
                       fill_color=itype_t, line_color=itype_t, visible=False)
            instr_items.append(LegendItem(label=name, renderers=[r]))

    ev_items = [
        LegendItem(
            label=ev_name,
            renderers=[marker_renderers[ev_name]],
        )
        for ev_name in marker_renderers
    ]

    if instr_items or ev_items:
        legend = Legend(
            items=instr_items + ev_items,
            location="center_right",
            orientation="vertical",
            click_policy="hide",
            title="Legend",
            label_text_font_size="8pt",
            spacing=1,
            margin=10,
            padding=5,
            glyph_height=15,
            glyph_width=15,
        )
        p.add_layout(legend, "right")

    return p


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Side-by-side SM-level Gantt: two MegaKernel runs"
    )
    parser.add_argument("--left",         type=Path, required=True,
                        help="First megakernel timing pkl")
    parser.add_argument("--left-label",   type=str,  default="Run A")
    parser.add_argument("--right",        type=Path, required=True,
                        help="Second megakernel timing pkl")
    parser.add_argument("--right-label",  type=str,  default="Run B")
    parser.add_argument("--depth",        type=int,  default=2,
                        help="k-slots (vertical sub-lanes per SM row, default 2)")
    parser.add_argument("--no-events",    action="store_true",
                        help="Omit sub-instruction event markers")
    parser.add_argument("--shared-x",     action="store_true",
                        help="Link x-axes for synchronised pan/zoom")
    parser.add_argument("--output",       type=Path, default=Path("mk_compare.html"))
    args = parser.parse_args()

    show_events = not args.no_events

    print(f"Building left chart  ({args.left_label}) ...")
    left_fig = make_mk_figure(args.left,  args.left_label,
                              k_slots=args.depth, show_events=show_events)

    print(f"Building right chart ({args.right_label}) ...")
    right_fig = make_mk_figure(args.right, args.right_label,
                               k_slots=args.depth, show_events=show_events,
                               x_range=left_fig.x_range if args.shared_x else None)

    layout = row(left_fig, right_fig)
    save(layout, filename=str(args.output),
         title=f"{args.left_label} vs {args.right_label}", resources=CDN)
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
