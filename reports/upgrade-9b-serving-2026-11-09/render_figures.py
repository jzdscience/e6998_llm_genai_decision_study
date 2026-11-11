#!/usr/bin/env python3
"""Render the report figure from the grouped table (summary.csv) built from accepted receipts.

  .venv-lab/bin/python reports/upgrade-9b-serving-2026-11-09/render_figures.py [--tables DIR] [--out DIR]

--tables is where summary.csv lives (default: this folder); --out receives
figures/correct_goodput.{png,svg} (default: this folder). With the Matplotlib version pinned in
paper/requirements-figures.txt the outputs are byte-for-byte reproducible.
"""
import argparse
import csv
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "decision-study-mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
RATES = (10, 20, 40, 80)
# Validated categorical slots (dataviz reference palette, --pairs all on white): blue, green, yellow.
# Yellow sits below 3:1 contrast, so every series also has its own marker and a legend entry.
# The last field is the offset inside a tested-rate group; it only separates the markers.
SERIES = (
    ("startlux4b", "StartLux-4B direct (GPU share 0.40)", "#2a78d6", "o", -0.22),
    ("startlux9b", "StartLux-9B direct (0.50)", "#008300", "s", 0.0),
    ("cascade", "StartLux-4B → Qwen3.5-9B cascade (0.40 + 0.50)", "#eda100", "D", 0.22),
)
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5, "legend.fontsize": 8.5, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.7, "svg.hashsalt": "decision-study",
    "svg.fonttype": "none", "savefig.facecolor": "white",
})


def load(tables):
    rows = {}
    with (tables / "summary.csv").open() as handle:
        for r in csv.DictReader(handle):
            rows[(r["condition"], int(r["rate_per_second"]), int(r["slo_ms"]))] = r
    if len(rows) != 24:
        raise SystemExit("summary.csv must hold 3 deployments x 4 rates x 2 SLOs")
    return rows


def panel(ax, rows, slo, label_endpoints=True):
    # Each tested rate is one categorical group; a short grey bar marks its offered rate.
    for group, rate in enumerate(RATES):
        ax.plot([group - 0.36, group + 0.36], [rate, rate], color=GRID, linewidth=1.8, zorder=0)
    ax.text(-0.36, 11.3, "offered rate", ha="left", va="bottom", fontsize=7.5, color=MUTED)
    for key, label, color, marker, offset in SERIES:
        for group, rate in enumerate(RATES):
            r = rows[(key, rate, slo)]
            x = group + offset
            values = sorted(float(r[f"correct_goodput_r{k}"]) for k in range(3))
            median = float(r["correct_goodput_median"])
            if median != values[1]:
                raise SystemExit("median must be the middle repeat")
            # With three repeats, the whisker ends (min, max) and the marker (median) are the three measurements.
            ax.plot([x, x], [values[0], values[2]], color=color, linewidth=1.0, zorder=2)
            ax.scatter([x, x], [values[0], values[2]], s=10, color=color, linewidths=0, zorder=3)
            passed = r["passes_99pct_all_repeats"] == "True"
            ax.scatter([x], [median], s=48, marker=marker, zorder=4, linewidths=1.3,
                       facecolor=color if passed else "white", edgecolor=color)
        if label_endpoints:
            r = rows[(key, 80, slo)]
            median = float(r["correct_goodput_median"])
            text = f"{median:.1f}" if median >= 10 else f"{median:.2f}"
            ax.annotate(text, (len(RATES) - 1 + offset, median), xytext=(7, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=8, color=INK)
    ax.set_yscale("log")
    ax.set_xlim(-0.55, len(RATES) - 0.25)
    ax.set_ylim(0.1, 110)
    ax.set_xticks(range(len(RATES)), [str(r) for r in RATES])
    ax.tick_params(axis="x", length=0)
    ax.set_yticks([0.1, 1, 10, 100], ["0.1", "1", "10", "100"])
    ax.minorticks_off()
    ax.set_xlabel("Tested offered load, ANLI test inputs (requests/s)")
    ax.set_ylabel(f"Correct answers within {slo:,} ms per s")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", type=Path, default=HERE)
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args()
    rows = load(args.tables)
    target = args.out / "figures"
    target.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6), gridspec_kw={"wspace": 0.28})
    for ax, slo, name in zip(axes, (500, 1000), ("500 ms SLO (primary)", "1,000 ms SLO (secondary)")):
        panel(ax, rows, slo)
        ax.set_title(name, loc="left", fontsize=10.5, color=INK, fontweight="bold")
    fig.text(0.07, 1.05, "Larger front model or a 9B fallback, under load", fontsize=15, fontweight="bold", color="#172f46")
    fig.text(0.07, 0.985, "ANLI test (3,200 inputs, retrospective) · one L40S · 3 repeats × 120 s per tested rate · "
             "requests arriving in the 120 s window and answered correctly within their own SLO, ÷ 120 s", fontsize=9.5, color=INK2)
    handles = [Line2D([0], [0], color=c, marker=m, markersize=7, linewidth=0) for _, _, c, m, _ in SERIES]
    handles += [Line2D([0], [0], color=INK2, marker="o", markersize=7, linewidth=0, markerfacecolor="white", markeredgecolor=INK2)]
    labels = [s[1] for s in SERIES] + ["open marker: at least one repeat below 99% on time"]
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False,
               columnspacing=2.0, handletextpad=0.5)
    fig.text(0.07, -0.205, "Each group is one tested rate; the deployments are offset inside it only for legibility, and the grey bar "
             "marks the offered rate. With three repeats, the large marker (median)\nand the two whisker ends (min, max) are the "
             "three measured values. A marker is filled when every repeat answered at least 99% of offers within the SLO; no other "
             "rate was measured.\nStartLux-4B at 20/s keeps its r0 repeat (96.9% on time at 500 ms, 97.2% at 1,000 ms), whose 78 "
             "late answers came from requests scheduled within a 3.6 s span. GPU shares differ by deployment.",
             fontsize=8.2, color=INK2, va="top", linespacing=1.45)
    fig.savefig(target / "correct_goodput.png", dpi=200, bbox_inches="tight", pad_inches=0.15)
    fig.savefig(target / "correct_goodput.svg", bbox_inches="tight", pad_inches=0.15, metadata={"Date": None})
    plt.close(fig)
    print("wrote", target / "correct_goodput.png", "and .svg")


if __name__ == "__main__":
    main()
