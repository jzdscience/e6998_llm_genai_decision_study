#!/usr/bin/env python3
"""Render the paper's data figures from evidence files tracked in this repository.

Run from the repository root:  .venv-lab/bin/python paper/figures/make_figures.py [--out DIR]
Reads only tracked, already-audited summaries; never touches raw evidence or models.
Outputs PDF (TrueType fonts, for ACL) and PNG previews next to this script.
"""
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "decision-study-mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
if len(sys.argv) > 2 and sys.argv[1] == "--out":
    OUT = Path(sys.argv[2]).resolve()
    OUT.mkdir(parents=True, exist_ok=True)

# Reference palette (dataviz skill), validated on a white surface: categorical slots
# 1-4 pass CVD/normal-vision gates; aqua and yellow sit below 3:1 contrast, so every
# series is also direct-labelled and carries its own marker shape.
BLUE, ORANGE, AQUA, YELLOW, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
ENTITY = {"startlux4b": BLUE, "qwen4b": ORANGE, "minilm1600": AQUA, "cascade": YELLOW}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white",
})


def load(rel):
    return json.loads((ROOT / rel).read_text())


def style(ax, grid_axis="x"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def save(fig, name):
    # No creation date in the PDF, so re-running the script reproduces the tracked files byte for byte.
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02, metadata={"CreationDate": None})
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def policy_rows():
    study = load("reports/decision-lab-2026-11-04/study.json")["experiments"]
    v2 = {p["id"]: p for p in study["v2"]["policies"]}
    v3 = {p["id"]: p for p in study["v3"]["policies"]}
    return v2, v3


def fig_rescue_harm():
    v2, v3 = policy_rows()
    # V1 SQuAD test, StartLux front with refitted 5% risk gate: 125 deferrals,
    # 29 rescued, 24 harmed of 2,000 (reports/formal-2026-11-02/RESULTS_ZH.md, section 3).
    rows = [("header", "SQuAD V1 test (n = 2,000)"),
            ("bar", "Risk gate (5% target)", 29, 24, 2000)]
    rows.append(("header", "SQuAD fresh holdout (n = 4,000), 10% budget"))
    for key, label in (("confidence", "Confidence"), ("class_conditional", "Class-conditional"), ("gain", "Gain")):
        p = v2[key]
        rows.append(("bar", label, p["rescue"], p["harm"], p["n"]))
    rows.append(("header", "ANLI test (n = 3,200), Qwen3.5-9B, 10% cap"))
    for key, label in (("qwen9b/confidence", "Confidence"), ("qwen9b/class_conditional", "Class-conditional"),
                       ("qwen9b/positive_gain", "Positive gain")):
        p = v3[key]
        rows.append(("bar", label, p["rescue"], p["harm"], p["n"]))

    fig, ax = plt.subplots(figsize=(3.05, 2.55))
    y, ticks, labels = 0.0, [], []
    for row in rows:
        if row[0] == "header":
            y -= 0.35
            ax.text(-6.3, y, row[1], ha="left", va="center", fontsize=6.5, color=INK2)
            y -= 0.85
            continue
        _, label, rescue, harm, n = row
        r, h = 100 * rescue / n, 100 * harm / n
        ax.barh(y, r, height=0.55, color=BLUE, linewidth=0)
        ax.barh(y, -h, height=0.55, color=RED, linewidth=0)
        note = "no deferrals" if rescue == harm == 0 else "net " + f"{rescue - harm:+d}".replace("-", "\u2212")
        ax.text(5.55, y, note, ha="left", va="center", fontsize=6.3, color=INK)
        ticks.append(y)
        labels.append(label)
        y -= 1.0
    ax.set_yticks(ticks, labels)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(y + 0.3, 0.45)
    ax.axvline(0, color=AXIS, linewidth=0.6)
    ax.set_xlim(-6.3, 7.6)
    ax.set_xticks([-6, -4, -2, 0, 2, 4, 6], ["6", "4", "2", "0", "2", "4", "6"])
    ax.set_xlabel("% of test inputs, harmed (left) and rescued (right)")
    style(ax)
    ax.spines["left"].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=RED)]
    ax.legend(handles, ["Rescued (front wrong, fallback right)", "Harmed (front right, fallback wrong)"],
              loc="lower center", bbox_to_anchor=(0.42, 1.0), ncol=1, frameon=False, handlelength=1.0)
    save(fig, "fig_rescue_harm")


def fig_budget_transfer():
    v2, v3 = policy_rows()
    rows = [("header", "SQuAD fresh holdout, 10% budget")]
    for key, label in (("confidence", "Confidence"), ("class_conditional", "Class-conditional"), ("gain", "Gain")):
        rows.append(("row", label, 100 * v2[key]["validation_deferral_rate"], 100 * v2[key]["deferral_rate"]))
    rows.append(("header", "ANLI test, Qwen3.5-9B, 10% cap"))
    for key, label in (("qwen9b/confidence", "Confidence"), ("qwen9b/class_conditional", "Class-conditional"),
                       ("qwen9b/fixed_gain", "Fixed gain"), ("qwen9b/positive_gain", "Positive gain")):
        rows.append(("row", label, 100 * v3[key]["validation_deferral_rate"], 100 * v3[key]["deferral_rate"]))

    fig, ax = plt.subplots(figsize=(3.05, 2.15))
    y, ticks, labels = 0.0, [], []
    for row in rows:
        if row[0] == "header":
            y -= 0.35
            ax.text(-0.5, y, row[1], ha="left", va="center", fontsize=6.5, color=INK2)
            y -= 0.85
            continue
        _, label, val, hold = row
        ax.plot([val, hold], [y, y], color=AXIS, linewidth=1.2, zorder=1)
        ax.scatter([val], [y], s=22, facecolor="white", edgecolor=MUTED, linewidth=1.0, zorder=2)
        ax.scatter([hold], [y], s=26, color=BLUE, edgecolor="white", linewidth=0.8, zorder=3)
        ax.text(max(val, hold) + 0.9, y, f"{hold:.1f}%", ha="left", va="center", fontsize=6.3, color=INK)
        ticks.append(y)
        labels.append(label)
        y -= 1.0
    ax.set_yticks(ticks, labels)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(y + 0.3, 0.45)
    ax.set_xlim(-0.5, 28.5)
    ax.set_xticks([0, 5, 10, 15, 20, 25])
    ax.set_xlabel("Share of inputs deferred to the fallback (%)")
    style(ax)
    ax.spines["left"].set_visible(False)
    handles = [ax.scatter([], [], s=22, facecolor="white", edgecolor=MUTED, linewidth=1.0),
               ax.scatter([], [], s=26, color=BLUE, edgecolor="white", linewidth=0.8)]
    ax.legend(handles, ["Validation (threshold chosen here)", "Test (same frozen threshold)"],
              loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, frameon=False, handletextpad=0.3,
              columnspacing=1.0)
    save(fig, "fig_budget_transfer")


def series_style(series):
    return [(label, ENTITY[key], marker) for key, label, marker in series]


def line_handles(styles):
    from matplotlib.lines import Line2D
    return [Line2D([0], [0], color=c, marker=m, markersize=4.2, markeredgecolor="white",
                   markeredgewidth=0.6, linewidth=1.3, label=l) for l, c, m in styles]


def fig_serving():
    agg = {}
    with open(ROOT / "reports/followups-2026-11-06/evidence/service/aggregates.csv") as fh:
        for r in csv.DictReader(fh):
            agg[(r["condition"], int(float(r["rate"])))] = {
                "median": float(r["correct_goodput_per_second_median"]),
                "min": float(r["correct_goodput_per_second_min"]),
                "max": float(r["correct_goodput_per_second_max"])}
    v2 = load("reports/v2-complete-2026-11-03/chart_data.json")["all_rounds"]

    fig, (a, b) = plt.subplots(1, 2, figsize=(6.3, 2.15), gridspec_kw={"wspace": 0.42})
    rates = [5, 25, 100]
    a.plot([4, 120], [4, 120], color=GRID, linewidth=1.0, zorder=0)
    a.text(118, 132, "offered rate", ha="right", va="bottom", fontsize=6, color=MUTED)
    series = [("minilm1600", "MiniLM-1600 (CPU)", "s"), ("startlux4b", "StartLux-4B (GPU)", "o"),
              ("qwen4b", "Qwen3.5-4B (GPU)", "^")]
    for key, label, marker in series:
        med = [agg[(key, r)]["median"] for r in rates]
        lo = [m - agg[(key, r)]["min"] for m, r in zip(med, rates)]
        hi = [agg[(key, r)]["max"] - m for m, r in zip(med, rates)]
        a.errorbar(rates, med, yerr=[lo, hi], color=ENTITY[key], marker=marker, markersize=4.2,
                   markeredgecolor="white", markeredgewidth=0.6, linewidth=1.3, capsize=1.6,
                   elinewidth=0.8, label=label)
    a.annotate("93.4", (100, agg[("minilm1600", 100)]["median"]), xytext=(4, 0), textcoords="offset points",
               ha="left", va="center", fontsize=6.3, color=INK)
    a.annotate("0.46", (100, agg[("startlux4b", 100)]["median"]), xytext=(4, 2), textcoords="offset points",
               ha="left", va="center", fontsize=6.3, color=INK)
    a.annotate("0.09", (100, agg[("qwen4b", 100)]["median"]), xytext=(4, -2), textcoords="offset points",
               ha="left", va="center", fontsize=6.3, color=INK)
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlim(4, 160)
    a.set_ylim(0.05, 200)
    a.set_xticks(rates, ["5", "25", "100"])
    a.set_yticks([0.1, 1, 10, 100], ["0.1", "1", "10", "100"])
    a.minorticks_off()
    a.set_xlabel("Offered load, CLINC (requests/s)")
    a.set_ylabel("On-time correct responses per s")
    style(a, "y")
    a.legend(handles=line_handles(series_style(series)), loc="lower left", frameon=False, handlelength=1.8)

    cond = [("local_only", "StartLux-4B alone", BLUE, "o"), ("selected_cascade", "Gain cascade to Qwen3.5-4B", YELLOW, "D"),
            ("qwen_only", "Qwen3.5-4B alone", ORANGE, "^")]
    v2rates = [10, 25, 40]
    b.plot([8, 48], [8, 48], color=GRID, linewidth=1.0, zorder=0)
    for key, label, color, marker in cond:
        pts = {float(r["requested_rate"]): r for r in v2 if r["condition"] == key}
        mean = [float(pts[float(x)]["goodput_per_second_mean"]) for x in v2rates]
        lo = [m - float(pts[float(x)]["goodput_per_second_min"]) for m, x in zip(mean, v2rates)]
        hi = [float(pts[float(x)]["goodput_per_second_max"]) - m for m, x in zip(mean, v2rates)]
        b.errorbar(v2rates, mean, yerr=[lo, hi], color=color, marker=marker, markersize=4.2,
                   markeredgecolor="white", markeredgewidth=0.6, linewidth=1.3, capsize=1.6,
                   elinewidth=0.8, label=label)
        b.annotate(f"{mean[-1]:.1f}", (40, mean[-1]), xytext=(4, 0), textcoords="offset points",
                   ha="left", va="center", fontsize=6.3, color=INK)
    b.set_xscale("log")
    b.set_yscale("log")
    b.set_xlim(8, 55)
    b.set_ylim(0.5, 100)
    b.set_xticks(v2rates, ["10", "25", "40"])
    b.set_yticks([1, 10, 100], ["1", "10", "100"])
    b.text(46, 51, "offered rate", ha="right", va="bottom", fontsize=6, color=MUTED)
    b.minorticks_off()
    b.set_xlabel("Offered load, SQuAD (requests/s)")
    b.set_ylabel("On-time responses per s")
    style(b, "y")
    b.legend(handles=line_handles([(label, color, marker) for _, label, color, marker in cond]),
             loc="lower left", frameon=False, handlelength=1.8)
    a.text(0.0, 1.03, "(a)", transform=a.transAxes, fontsize=7, color=INK)
    b.text(0.0, 1.03, "(b)", transform=b.transAxes, fontsize=7, color=INK)
    save(fig, "fig_serving")


if __name__ == "__main__":
    fig_rescue_harm()
    fig_budget_transfer()
    fig_serving()
    print("wrote", sorted(p.name for p in OUT.glob("fig_*.pdf")))
