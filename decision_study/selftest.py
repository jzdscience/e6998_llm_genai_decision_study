"""Clearly labeled synthetic software verification, never research evidence."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .analysis import evaluate_experiment, fit_model
from .common import fingerprint, write_json, write_jsonl
from .data import BANKING_LABELS, BANKING_CRITERIA, CLINC_INSTRUCTIONS, SQUAD_LABELS, SQUAD_CRITERIA, SQUAD_INSTRUCTIONS


def fixture_data():
    examples, gold, predictions = [], {}, {m:[] for m in ("kev","startlux","qwen")}
    for task in ("clinc","squad"):
        labels = BANKING_LABELS if task == "clinc" else SQUAD_LABELS
        criteria = BANKING_CRITERIA if task == "clinc" else SQUAD_CRITERIA
        instruction = CLINC_INSTRUCTIONS if task == "clinc" else SQUAD_INSTRUCTIONS
        for split, count in (("calibration",96),("validation",96),("test",128),("stress",32 if task == "clinc" else 0)):
            for i in range(count):
                sid = f"fixture-{task}-{split}-{i:04d}"
                label = "oos" if split == "stress" else labels[i % len(labels)]
                examples.append({"sample_id":sid,"task":task,"split":split,"group_id":f"fixture-{task}-{split}-article-{i//8}",
                                 "state":f"SOFTWARE TEST INPUT {i}. This is not a benchmark question.",
                                 "instructions":instruction,"labels":labels,"criteria":criteria})
                gold[sid] = label
    for model in predictions:
        rng = np.random.default_rng({"kev":101,"startlux":202,"qwen":303}[model])
        for e in examples:
            if model == "qwen" and e["split"] not in {"test","stress"}:
                continue
            true = e["labels"].index(gold[e["sample_id"]])
            low = rng.random() < .35
            wrong = rng.random() < (.35 if low else .02)
            pred = (true + 1) % len(e["labels"]) if wrong else true
            probs = None
            if model != "qwen":
                confidence = (.55 if len(e["labels"]) == 2 else .3) if low else .97
                values = np.full(len(e["labels"]),(1-confidence)/(len(e["labels"])-1))
                values[pred] = confidence
                probs = dict(zip(e["labels"],values.tolist()))
            predictions[model].append({"sample_id":e["sample_id"],"model":model,"task":e["task"],"split":e["split"],
                "config_hash":fingerprint({"software_test_fixture":True,"model":model}),"status":"ok",
                "prediction":e["labels"][pred],"probabilities":probs,"latency_ms":None,"attempts":[],
                "error":None,"is_test_fixture":True})
    return examples, gold, predictions


def run_selftest(output_dir: Path) -> dict:
    from .plots import plot_results
    root = Path(output_dir) / "software_test_fixture"
    examples, gold, predictions = fixture_data()
    write_json(root/"NOTICE.json",{"evidence_type":"software_test_fixture","notice":"Artificial data and predictions test the software only. No GPU model was evaluated."})
    write_jsonl(root/"fixture_inputs.jsonl",examples)
    write_json(root/"fixture_gold.json",gold)
    for model, rows in predictions.items():
        write_jsonl(root/f"fixture_predictions_{model}.jsonl",rows)
    calibrations = {m:fit_model(examples,gold,predictions[m],m,root) for m in ("kev","startlux")}
    metrics = evaluate_experiment(examples,gold,predictions,calibrations,root,bootstrap_reps=100,random_reps=100)
    if metrics["evidence_type"] != "software_test_fixture":
        raise AssertionError("Fixture evidence labeling failed")
    figures = plot_results(root/"analysis",fixture=True)
    return {"status":"passed","evidence_type":"software_test_fixture","directory":str(root),
            "figures":len(figures),"notice":"CPU software self-test only. No real model predictions or timing results."}
