"""Publication-friendly charts computed only from the exported analysis artifacts."""
from __future__ import annotations

import json
from pathlib import Path

COLORS = {"kev": "#176B9B", "startlux": "#C45A20", "qwen": "#53674D"}
NAMES = {"kev": "Kev", "startlux": "StartLux", "qwen": "Qwen"}


def plot_results(analysis_dir, fixture=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    directory = Path(analysis_dir)
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    fixture = fixture or metrics.get("evidence_type") == "software_test_fixture"
    curves = json.loads((directory / "curves.json").read_text(encoding="utf-8"))
    figures = directory / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlesize": 12, "figure.dpi": 120, "savefig.dpi": 180,
                         "axes.grid": True, "grid.alpha": .18, "axes.axisbelow": True})
    outputs = []

    def save(fig, name, note):
        if fixture:
            fig.text(.5, .995, "SOFTWARE TEST FIXTURE — NOT MODEL RESULTS", ha="center", va="top", color="#A02515", fontsize=11, weight="bold")
        fig.text(.02, .015, note, fontsize=8, va="bottom", wrap=True)
        fig.tight_layout(rect=(0, .09, 1, .97))
        for suffix in ("png", "pdf"):
            path = figures / f"{name}.{suffix}"
            fig.savefig(path, bbox_inches="tight")
            outputs.append(str(path))
        plt.close(fig)

    for task, workloads in metrics["datasets"].items():
        main = workloads["main"]
        chart = curves[task]["main"]
        fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
        models = ["kev", "startlux", "qwen"]
        values = [main["direct"][m]["accuracy"] for m in models]
        axes[0].bar([NAMES[m] for m in models], values, color=[COLORS[m] for m in models])
        for i, (model, value) in enumerate(zip(models, values)):
            ci = main["uncertainty"]["intervals"][f"direct/{model}"]["accuracy"]
            if ci["lower"] is not None and ci["upper"] is not None:
                axes[0].vlines(i, ci["lower"], ci["upper"], color="#222222", linewidth=1.2)
                axes[0].hlines([ci["lower"], ci["upper"]], i - .07, i + .07, color="#222222", linewidth=1.2)
            axes[0].annotate(f"{value:.1%}", (i, value), xytext=(0, 5), textcoords="offset points", ha="center")
        axes[0].set(ylim=(0, 1.08), ylabel="Accuracy", title=f"{task.upper()}: direct decisions")
        categories = [("both_correct", "Both correct", "#477C69"),
                      ("local_only_correct", "Local only", "#176B9B"),
                      ("qwen_only_correct", "Qwen only", "#D49B43"),
                      ("both_wrong", "Both wrong", "#B65757")]
        bottom = np.zeros(2)
        for field, label, color in categories:
            values = np.array([main["complementarity"][m][field] / main["n"] for m in models[:2]])
            axes[1].bar([NAMES[m] for m in models[:2]], values, bottom=bottom, label=label, color=color)
            bottom += values
        axes[1].set(ylim=(0, 1), ylabel="Fraction of all test inputs", title="Decision/Qwen complementarity")
        axes[1].legend(loc="upper center", bbox_to_anchor=(.5, -.12), ncol=2, fontsize=8)
        unit = "article-cluster" if task == "squad" else "class-stratified"
        save(fig, f"{task}_accuracy_complementarity", f"n={main['n']}; failures count as incorrect. Bars: 95% paired {unit} bootstrap, conditional on frozen systems.")

        fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
        axes[0].plot([0, 1], [0, 1], color="#777777", linewidth=1, linestyle=":", label="Perfect calibration")
        for name, bins in chart["reliability"].items():
            model, condition = name.split("/")
            populated = [b for b in bins if b["n"]]
            style = "-" if condition == "refitted" else "--"
            label = f"{NAMES[model]} {condition}"
            axes[0].plot([b["confidence"] for b in populated], [b["accuracy"] for b in populated],
                         marker="o", markersize=3, color=COLORS[model], linestyle=style, label=label)
            pts = [p for p in chart["risk_coverage"][name] if p["risk"] is not None]
            axes[1].plot([p["coverage"] for p in pts], [p["risk"] for p in pts],
                         color=COLORS[model], linestyle=style, marker=".", markersize=2.5, label=label)
            route = main["routes"][name]
            if route["selective_risk"] is not None:
                axes[1].scatter([route["coverage"]], [route["selective_risk"]], color=COLORS[model],
                                marker="o" if condition == "refitted" else "x", s=24)
        axes[0].set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean confidence in bin", ylabel="Observed accuracy", title="Reliability (15 equal-width bins)")
        axes[1].axhline(.05, color="#777777", linestyle=":", linewidth=1, label="5% target")
        axes[1].set(xlim=(0, 1), ylim=(0, 1), xlabel="Local coverage / all inputs", ylabel="Error among accepted", title="Risk–coverage; larger marks: frozen gates")
        for ax in axes:
            ax.legend(fontsize=7, loc="best")
        save(fig, f"{task}_calibration_risk", "Curves describe held-out predictions; gates were selected on validation. Empty bins are omitted. Points preserve score ties; the 5% line is not a guarantee.")

        fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
        for ax, model in zip(axes, models[:2]):
            for condition in ("released", "refitted"):
                pts = chart["accuracy_deferral"][f"{model}/{condition}"]
                ax.plot([p["deferral_fraction"] for p in pts], [p["confidence_accuracy"] for p in pts],
                        color=COLORS[model], linestyle="--" if condition == "released" else "-", marker=".", markersize=2.5, label=f"Confidence: {condition}")
                if condition == "refitted":
                    pts = chart["routing_reference"][model]
                    ax.plot([p["deferral_fraction"] for p in pts], [p["random_expected_accuracy"] for p in pts],
                            color="#777777", linestyle=":", label="Random: exact expectation")
                    ax.plot([p["deferral_fraction"] for p in pts], [p["oracle_accuracy"] for p in pts],
                            color="#6E5594", linestyle="-.", label="Hindsight oracle")
                r = main["routes"][f"{model}/{condition}"]
                ax.scatter([r["fallback_fraction"]], [r["cascade_accuracy"]], color=COLORS[model],
                           marker="o" if condition == "refitted" else "x", s=26)
            ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Fraction deferred to Qwen", ylabel="Cascade accuracy", title=f"{NAMES[model]} → Qwen")
            ax.legend(fontsize=7, loc="best")
        save(fig, f"{task}_routing", "Same real Qwen predictions; exact matched quotas include mandatory failure deferrals. Oracle uses gold and is not deployable.")

        fig, axes = plt.subplots(1, 3, figsize=(11, 4.1))
        names = list(main["probability"])
        for ax, metric in zip(axes, ("nll", "brier", "ece")):
            vals = [main["probability"][name][metric] for name in names]
            for i, (name, value) in enumerate(zip(names, vals)):
                model, condition = name.split("/")
                if value is not None:
                    ax.bar(i, value, color=COLORS[model], alpha=.95 if condition == "refitted" else .45,
                           hatch=None if condition == "refitted" else "//")
            ax.set_xticks(range(len(names)), [n.replace("/", "\n") for n in names], fontsize=8)
            ax.set(title=metric.upper(), ylabel="Lower is better", ylim=(0, None))
        save(fig, f"{task}_probability_quality", "Valid complete probability distributions only; exact valid/invalid denominators are in probability_metrics.csv. Qwen has no comparable probability arm.")

    if "clinc" in curves and "prevalence" in curves["clinc"]:
        fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
        for name, points in curves["clinc"]["prevalence"]["series"].items():
            bits = name.split("/")
            model = bits[1]
            if bits[0] == "cascade" and bits[-1] != "refitted":
                continue
            label = f"{NAMES[model]} direct" if bits[0] == "direct" else f"{NAMES[model]} → Qwen"
            axes[0].plot([p["oos_fraction"] for p in points], [p["accuracy"] for p in points],
                         color=COLORS[model], linestyle=":" if bits[0] == "direct" else "-", label=label)
            if bits[0] == "cascade":
                axes[1].plot([p["oos_fraction"] for p in points], [p["coverage"] for p in points], color=COLORS[model], label=label)
        for ax in axes:
            ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Reweighted OOS prevalence")
            ax.legend(fontsize=7, loc="best")
        axes[0].set(ylabel="Accuracy", title="Fixed conditional pools: accuracy")
        axes[1].set(ylabel="Local coverage", title="Frozen refitted gates: coverage")
        p = curves["clinc"]["prevalence"]
        save(fig, "clinc_prevalence", f"Reweighting {p['banking_n']} banking + {p['oos_n']} OOS inputs. No refitting; curve points are dependent and do not add observations.")
    return outputs
