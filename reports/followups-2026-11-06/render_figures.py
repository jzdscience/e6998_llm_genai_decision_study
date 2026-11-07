"""Render the two audited study figures from byte-preserved evaluation copies.

Run after reviewing RESULTS_ZH.md and the audit receipts. No model execution,
fitting, or modification of the source experiment evidence is performed.
"""
from collections import defaultdict
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "decision-study-mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)
COLORS = {"minilm400": "#50958A", "minilm1600": "#006D77",
          "startlux4b": "#CC713B", "qwen4b": "#77508C",
          "startlux9b": "#B74B19", "qwen9b": "#492A61"}
LABELS = {"minilm400": "MiniLM400 · CPU", "minilm1600": "MiniLM1600 · CPU",
          "startlux4b": "StartLux4B", "qwen4b": "Qwen4B",
          "startlux9b": "StartLux9B", "qwen9b": "Qwen9B"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.titlesize": 13, "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.labelcolor": "#263442", "text.color": "#263442",
                     "xtick.color": "#52616B", "ytick.color": "#52616B",
                     "svg.fonttype": "none", "savefig.facecolor": "white"})


def save(fig, name):
    fig.savefig(FIGURES / (name + ".png"), dpi=180, bbox_inches="tight")
    fig.savefig(FIGURES / (name + ".svg"), bbox_inches="tight")
    plt.close(fig)


def service_figure():
    evaluation = json.loads((ROOT / "service-evaluation.json").read_text())
    assert evaluation["status"] == "COMPLETE" and len(evaluation["cells"]) == 60
    conditions = ["minilm400", "minilm1600", "startlux4b", "qwen4b"]
    rates = [5, 25, 100]
    groups = defaultdict(list)
    for cell in evaluation["cells"]:
        groups[(cell["condition"], cell["rate"])].append(cell)
    assert all(len(groups[(c, r)]) == 5 for c in conditions for r in rates)
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.2), gridspec_kw={"width_ratios": [1.12, 1]})
    fig.subplots_adjust(top=.76, bottom=.32, wspace=.42)
    fig.text(.06, .95, "When a successful answer arrives too late", fontsize=20, weight="bold")
    fig.text(.06, .895, "CLINC600 · 500 ms SLO · 4 direct services · 5 repetitions per fixed offered rate", fontsize=11)
    ax = axes[0]
    x, width = np.arange(3), .19
    for index, condition in enumerate(conditions):
        values = [[c["correct_goodput_per_second"] for c in groups[(condition, r)]] for r in rates]
        middle = np.median(values, axis=1)
        yerr = [middle - np.min(values, axis=1), np.max(values, axis=1) - middle]
        positions = x + (index - 1.5) * width
        ax.bar(positions, middle, width, color=COLORS[condition], label=LABELS[condition],
               yerr=yerr, capsize=2, error_kw={"elinewidth": 1, "ecolor": "#263442"})
        for pos, value, high in zip(positions, middle, np.max(values, axis=1)):
            ax.text(pos, high + 1.7 + (index % 2)*1.6, f"{value:.2f}" if value < 1 else f"{value:.1f}",
                    ha="center", va="bottom", fontsize=8.5)
    ax.set_title("Timely and correct outputs")
    ax.set_ylabel("Correct responses within 500 ms / second")
    ax.set_xticks(x, [str(r) for r in rates])
    ax.set_xlabel("Offered rate (requests / second)")
    ax.set_ylim(0, 106)
    ax.yaxis.grid(True, color="#E5E9ED")
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(-.03, -.23), ncol=2, frameon=False, fontsize=9)
    ax = axes[1]
    high_load = [groups[(c, 100)] for c in conditions]
    fields = ["Timely & correct", "Timely & wrong", "Successful, >500 ms", "Client rejected"]
    palette = ["#176B50", "#A9D2BF", "#D5DCE3", "#536675"]
    matrix = []
    for rows in high_load:
        total = sum(r["offered"] for r in rows)
        timely = sum(r["met_slo"] for r in rows)
        correct = sum(r["correct_met_slo"] for r in rows)
        success = sum(r["successful"] for r in rows)
        rejected = sum(r["rejected"] for r in rows)
        assert success + rejected == total and all(r["failed_admitted"] == 0 for r in rows)
        matrix.append(np.array([correct, timely-correct, success-timely, rejected])*100/total)
    matrix = np.asarray(matrix)
    assert np.allclose(matrix.sum(axis=1), 100)
    left = np.zeros(4)
    for i, (field, color) in enumerate(zip(fields, palette)):
        ax.barh(np.arange(4), matrix[:, i], left=left, color=color, height=.58, label=field)
        for row, value in enumerate(matrix[:, i]):
            if value > 6:
                ax.text(left[row] + value/2, row, f"{value:.1f}%", ha="center", va="center",
                        color="white" if i in (0, 3) else "#263442", fontsize=10)
        left += matrix[:, i]
    ax.set_title("Every offer at 100 requests / second")
    ax.set_yticks(np.arange(4), [LABELS[c] for c in conditions])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of all offers (%) · pooled 5 repetitions")
    ax.legend(loc="upper left", bbox_to_anchor=(-.03, -.23), ncol=2, frameon=False, fontsize=9)
    fig.text(.06, .035, "Left: medians; whiskers = observed min–max, not confidence intervals. "
             "Each bar uses a 180 s denominator.\nTwo allocations; all 100/s comparisons are paired within allocation. "
             "Repeated CLINC inputs are not independent accuracy samples.", fontsize=9, color="#52616B")
    save(fig, "service_correct_goodput")


def size_figure():
    evaluation = json.loads((ROOT / "size-evaluation.json").read_text())
    assert evaluation["status"] == "complete" and evaluation["n"] == 3200
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5), gridspec_kw={"width_ratios": [1.05, 1]})
    fig.subplots_adjust(top=.75, bottom=.25, left=.13, right=.96, wspace=.61)
    fig.text(.06, .94, "StartLux9B gains 2.50 points over StartLux4B", fontsize=19, weight="bold")
    fig.text(.06, .88, "ANLI · the same 3,200 inputs · direct models · retrospective size control", fontsize=11)
    ax = axes[0]
    order = ["qwen4b", "qwen9b", "startlux4b", "startlux9b"]
    for i, model in enumerate(order):
        result = evaluation["direct"][model]
        ax.barh(i, 100*result["accuracy"], color=COLORS[model], height=.58)
        ax.text(100*result["accuracy"]+1, i, f"{100*result['accuracy']:.2f}%", va="center", fontsize=10)
    ax.set_yticks(range(4), [LABELS[m] for m in order])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Question-weighted accuracy (%)")
    ax.set_title("Direct accuracy")
    ax.xaxis.grid(True, color="#E5E9ED")
    ax.set_axisbelow(True)
    ax = axes[1]
    comparisons = ["startlux9b_minus_startlux4b", "startlux9b_minus_qwen9b"]
    for y, key in enumerate(comparisons):
        d = evaluation["comparisons_descriptive"][key]["difference"]
        estimate, low, high = [100*d[k] for k in ("estimate", "lower", "upper")]
        ax.errorbar(estimate, y, xerr=[[estimate-low], [high-estimate]], fmt="o", capsize=5,
                    markersize=7, color=COLORS["startlux9b"], elinewidth=2)
        ax.text(estimate, y+.24, f"+{estimate:.2f} pp  [{low:.2f}, {high:.2f}]",
                ha="center", va="top", fontsize=10)
    ax.axvline(0, color="#8A959F", linestyle="--", linewidth=1)
    ax.set_yticks([0, 1], ["vs StartLux4B", "vs Qwen9B"])
    ax.set_ylim(1.65, -.55)
    ax.set_xlim(-.7, 13.7)
    ax.set_xticks([0, 4, 8, 12])
    ax.set_xlabel("StartLux9B accuracy difference (percentage points)")
    ax.set_title("Paired differences · 95% intervals")
    ax.xaxis.grid(True, color="#E5E9ED")
    ax.set_axisbelow(True)
    fig.text(.06, .07, "Intervals: 2,000 paired premise-cluster bootstrap replicates; 2,854 normalized premises.\n"
             "Previously inspected test pool; no claim of an architecture-only effect, routing gain or 9B serving speed.",
             fontsize=9, color="#52616B")
    save(fig, "size_control_accuracy")


if __name__ == "__main__":
    service_figure()
    size_figure()
