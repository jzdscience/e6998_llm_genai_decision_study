"""Render separate report figures without changing the returned benchmark bundle."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/returned-benchmark-2026-11-02-130735/benchmark/summary.json"
OUT = Path(__file__).resolve().parent / "figures"
DEPLOYMENTS = ["kev_direct", "startlux_direct", "qwen_direct", "kev_to_qwen", "startlux_to_qwen"]
LABELS = ["Kev", "StartLux", "Qwen", "Kev → Qwen", "StartLux → Qwen"]
COLORS = ["#216B95", "#C47A24"]


def main():
    rows = json.loads(SOURCE.read_text())
    assert len(rows) == 60
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    for metric, filename, title, xlabel, xmax in [
        ("successful_inputs_per_second", "throughput", "Live serving throughput on one L40S",
         "Successful requests / second  ·  higher is better", 67),
        ("p95_ms", "request_latency_p95", "Measured request latency on one L40S",
         "Mean of three round-specific p95 values (ms)  ·  lower is better", 875),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.5), sharex=True, sharey=True)
        for ax, task in zip(axes, ("clinc", "squad")):
            for idx, concurrency in enumerate((1, 8)):
                values = []
                for deployment in DEPLOYMENTS:
                    group = [r for r in rows if (r["task"], r["deployment"], r["concurrency"])
                             == (task, deployment, concurrency)]
                    assert sorted(r["repeat"] for r in group) == [0, 1, 2]
                    values.append([r[metric] for r in group])
                values = np.asarray(values)
                means, lows, highs = values.mean(1), values.min(1), values.max(1)
                y = np.arange(len(DEPLOYMENTS)) + (idx - .5) * .34
                ax.barh(y, means, height=.30, color=COLORS[idx],
                        xerr=np.vstack((means-lows, highs-means)),
                        error_kw={"capsize": 2, "elinewidth": 1, "ecolor": "#333333"},
                        label=f"Concurrency {concurrency}", zorder=3)
                for ypos, value, high in zip(y, means, highs):
                    ax.text(high + xmax * .012, ypos, f"{value:.1f}", va="center", fontsize=9)
            ax.set_title("CLINC" if task == "clinc" else "SQuAD answerability", loc="left", weight="bold")
            ax.set_yticks(np.arange(len(DEPLOYMENTS)), LABELS)
            ax.set_xlim(0, xmax)
            ax.grid(axis="x", color="#DDE2E6", linewidth=.7, zorder=0)
            ax.tick_params(axis="y", length=0)
        axes[0].invert_yaxis()
        fig.suptitle(title, x=.07, ha="left", fontsize=18, weight="bold")
        fig.supxlabel(xlabel, y=.135, fontsize=11)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(.97, .99), ncol=2, frameon=False)
        note = ("Fixed 200 unique inputs per task; 3 rounds. Bars: arithmetic means; whiskers: observed min–max, not confidence intervals.\n"
                "Timing excludes service startup and warm-up; throughput uses complete measured-round wall time.")
        if metric == "p95_ms":
            note = ("Fixed 200 unique inputs per task; 3 rounds. Mean of round p95s, not a pooled p95; whiskers are min–max, not confidence intervals.\n"
                    "Request timer starts inside the worker; excludes the controller's pre-worker queue, service startup and warm-up.")
        fig.text(.07, .025, note, fontsize=9, color="#444444", linespacing=1.5)
        fig.subplots_adjust(left=.13, right=.96, bottom=.23, top=.84, wspace=.18)
        for suffix in ("png", "pdf"):
            fig.savefig(OUT / f"{filename}.{suffix}", dpi=180, facecolor="white")
        plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
