"""Software fixtures only. Real data preparation is covered by test_v2_data."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from decision_study.analysis import fit_model
from decision_study.adapters import decision_request, qwen_request
from decision_study.runner import _probability_comparison
from decision_study.common import (StudyError, file_hash, fingerprint, read_json,
                                  read_jsonl, write_json, write_jsonl)
from decision_study.storage import PredictionStore
from decision_study.v2 import study


def make_fixture(root: Path, *, reverse_val_gold=False, sealed_invalid=False):
    v1, out = root / "v1", root / "v2"
    labels = ["answerable", "unanswerable"]
    fitting, holdout, gold = [], [], {}
    rows = {"startlux": [], "qwen": []}
    for split, n in (("calibration", 40), ("validation", 20), ("holdout", 16)):
        for i in range(n):
            sid = f"fixture-{split}-{i}"
            predicted = labels[i % 2]
            x = {"sample_id": sid, "task": "squad", "split": split, "labels": labels,
                 "criteria": {k: k for k in labels}, "instructions": "Software fixture only",
                 "state": "x" * (10 + i * 13), "group_id": f"{split}-article-{i // 4}"}
            (holdout if split == "holdout" else fitting).append(x)
            y = predicted if i % 3 else labels[1 - i % 2]
            gold[sid] = labels[1 - labels.index(y)] if reverse_val_gold and split == "validation" else y
            c = .55 + .01 * i
            rows["startlux"].append({"sample_id": sid, "status": "ok", "prediction": predicted,
                                     "probabilities": {predicted: c, labels[1 - i % 2]: 1 - c},
                                     "is_test_fixture": True})
            rows["qwen"].append({"sample_id": sid, "status": "ok",
                                 "prediction": y if i % 4 else labels[1 - labels.index(y)],
                                 "probabilities": None, "is_test_fixture": True})
    fit_ids = {x["sample_id"] for x in fitting}
    fit_gold = {sid: gold[sid] for sid in fit_ids}
    dev = [{**fitting[0], "sample_id": "fixture-development", "split": "development",
            "group_id": "development-article"}]
    write_jsonl(v1 / "data/examples.jsonl", fitting + dev)
    write_json(v1 / "data/gold.json", fit_gold)

    def store(where, model, selected):
        config = {"model": model, "is_test_fixture": True, "runtime": {"fixture_runtime": model}}
        s = PredictionStore(where, model, config, spool_dir=root / "spool", shard_size=10)
        saved = [{**r, "model": model, "config_hash": fingerprint(config)} for r in selected]
        for r in saved:
            s.append(r)
        s.sync()
        return saved

    local_fit = store(v1, "startlux", [r for r in rows["startlux"] if r["sample_id"] in fit_ids])
    write_json(v1 / "predictions/qwen/config.json", {"model": "qwen", "is_test_fixture": True, "runtime": {"fixture_runtime": "qwen"}})
    store(out / "predictions-jobs/qwen_fit", "qwen", [r for r in rows["qwen"] if r["sample_id"] in fit_ids])
    for model in rows:
        store(out / f"predictions-jobs/holdout_{model}", model, [r for r in rows[model] if r["sample_id"] not in fit_ids])
    fit_model(fitting, fit_gold, local_fit, "startlux", v1)
    write_jsonl(out / "data/examples.jsonl", fitting + holdout)
    write_json(out / "data/fit_gold.json", fit_gold)
    write_json(out / "sealed/holdout_gold.json", {x["sample_id"]: gold[x["sample_id"]] for x in holdout})
    if sealed_invalid:
        (out / "sealed/holdout_gold.json").write_text("SEALED: not available to fitting")
    protocol = read_json(Path(__file__).resolve().parents[1] / "configs/v2_protocol.json")
    protocol["data"].update(qwen_fit_count=len(fitting), holdout_per_class=8)
    protocol["evaluation"]["bootstrap_reps"] = 40
    manifest = {"protocol": "solo-v2", "evidence_type": "software_test_fixture",
                "protocol_identity": {"configuration": protocol},
                "files": {p: file_hash(out / p) for p in ["data/examples.jsonl", "data/fit_gold.json", "sealed/holdout_gold.json"]}}
    manifest["manifest_hash"] = fingerprint(manifest)
    write_json(out / "data/manifest.json", manifest)
    for job, model, examples in (("qwen_fit", "qwen", fitting), ("holdout_qwen", "qwen", holdout),
                                  ("holdout_startlux", "startlux", holdout)):
        jobdir = out / "predictions-jobs" / job
        original = read_json(v1 / f"predictions/{model}/config.json")
        renderer = qwen_request if model == "qwen" else lambda e: decision_request(e, model)
        config = {"protocol": "solo-v2", "job": job, "model": model,
                  "data_manifest_hash": fingerprint(manifest), "sources": {"fixture_source": "version-1"},
                  "protocol_hash": fingerprint(protocol), "runtime": original["runtime"],
                  "v1_config_hash": fingerprint(original),
                  "request_hash": fingerprint([{"id": e["sample_id"], "request": renderer(e)} for e in examples]),
                  "expected_predictions": len(examples), "max_attempts": 2, "timeout_seconds": 120}
        h = fingerprint(config)
        write_json(jobdir / f"predictions/{model}/config.json", config)
        for path in (jobdir / f"predictions/{model}").glob("shard-*.json"):
            shard = read_json(path)
            shard["config_hash"] = h
            for row in shard["rows"]:
                row["config_hash"] = h
            shard["sha256"] = fingerprint(shard["rows"])
            write_json(path, shard)
        index_path = jobdir / f"predictions/{model}/index.json"
        index = read_json(index_path)
        index["config_hash"] = h
        write_json(index_path, index)
        serial = [{"sample_id": dev[0]["sample_id"], "status": "ok", "prediction": "answerable",
                   "probabilities": None if model == "qwen" else {"answerable": .8, "unanswerable": .2},
                   "is_test_fixture": True}]
        write_json(v1 / f"smoke/{model}.json", {"serial": serial})
        write_json(jobdir / "smoke.json", {"config_hash": h, "rows": serial,
                                           "checks": _probability_comparison(dev, serial, serial)})
        write_json(jobdir / "completion.json", {"job": job, "model": model, "config_hash": h,
                                                "count": len(examples), "failures": 0})
    return v1, out


class StudyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.v1, self.out = make_fixture(self.root)

        # Unit boundary: exercise study's own original-input checks and binding
        # checks while using small, explicitly marked software data. Production
        # data_v2.verify_* is independently tested on full source identities.
        def verify(out, sealed=False):
            out = Path(out)
            m = read_json(out / "data/manifest.json")
            study._checked_self_hash(m, "manifest_hash")
            for path, expected in m["files"].items():
                if sealed or not path.startswith("sealed/"):
                    if file_hash(out / path) != expected:
                        raise StudyError("Fixture data changed")
            return m

        def examples(out, role=None):
            rows = read_jsonl(Path(out) / "data/examples.jsonl")
            return [r for r in rows if (r["split"] == "holdout") == (role == "holdout")]

        self.patches = [patch.object(study.data_v2, "verify_fit", side_effect=verify),
                        patch.object(study.data_v2, "verify", side_effect=lambda p: verify(p, True)),
                        patch.object(study.data_v2, "load_examples", side_effect=examples),
                        patch.object(study.data_v2, "load_fit_gold", side_effect=lambda p: read_json(Path(p) / "data/fit_gold.json")),
                        patch.object(study, "_source_hashes", return_value={"fixture_source": "version-1"})]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.temp.cleanup()

    def test_fit_succeeds_without_reading_sealed_labels(self):
        # Deliberately not JSON. A leakage mutation that opens/parses this file
        # while fitting fails, whereas its prepared hash can still be bound.
        self.v1, self.out = make_fixture(self.root / "unreadable_seal", sealed_invalid=True)
        artifact = study.fit(self.v1, self.out)
        self.assertEqual(artifact["evidence_type"], "software_test_fixture")
        self.assertEqual(artifact["selected_policy"]["fit_count"], 40)
        self.assertEqual(artifact["selected_policy"]["decision"]["selection_n"], 20)
        self.assertFalse((self.out / "holdout_access.json").exists())

    def test_validation_labels_do_not_train_heads(self):
        first = study.fit(self.v1, self.out)
        v1b, outb = make_fixture(self.root / "changed_validation", reverse_val_gold=True)
        second = study.fit(v1b, outb)
        a = first["candidates_by_budget"]["0.1"]["policies"]
        b = second["candidates_by_budget"]["0.1"]["policies"]
        for family in ("class_conditional", "gain"):
            self.assertEqual(a[family]["local_head"], b[family]["local_head"])
        self.assertEqual(a["gain"]["qwen_head"], b["gain"]["qwen_head"])

    def test_unfrozen_evaluation_never_opens_holdout(self):
        with self.assertRaisesRegex(StudyError, "freeze"):
            study.evaluate(self.v1, self.out)
        self.assertFalse((self.out / "holdout_access.json").exists())

    def test_mutated_policy_and_source_are_rejected(self):
        artifact = study.fit(self.v1, self.out)
        changed = copy.deepcopy(artifact)
        changed["selected_name"] = "tampered"
        write_json(self.out / "policy_freeze.json", changed)
        with self.assertRaisesRegex(StudyError, "freeze_hash"):
            study.load_freeze(self.out)
        write_json(self.out / "policy_freeze.json", artifact)
        with patch.object(study, "_source_hashes", return_value={"fixture_source": "mutated"}):
            with self.assertRaisesRegex(StudyError, "source identity"):
                study.evaluate(self.v1, self.out)
        self.assertFalse((self.out / "holdout_access.json").exists())

    def test_job_smoke_and_runtime_provenance_are_verified(self):
        path = self.out / "predictions-jobs/qwen_fit/smoke.json"
        smoke = read_json(path)
        smoke["checks"]["squad"]["argmax_agreement"] = .5
        write_json(path, smoke)
        with self.assertRaisesRegex(StudyError, "smoke evidence failed"):
            study.fit(self.v1, self.out)
        self.assertFalse((self.out / "policy_freeze.json").exists())

    def test_fitting_prediction_mutation_rejected_even_with_recomputed_shard_hash(self):
        study.fit(self.v1, self.out)
        path = next((self.out / "predictions-jobs/qwen_fit/predictions/qwen").glob("shard-*.json"))
        shard = read_json(path)
        row = shard["rows"][0]
        row["prediction"] = "unanswerable" if row["prediction"] == "answerable" else "answerable"
        shard["sha256"] = fingerprint(shard["rows"])
        write_json(path, shard)
        with self.assertRaisesRegex(StudyError, "changed after policy freeze"):
            study.evaluate(self.v1, self.out)

    def test_evaluation_is_idempotent_and_binds_first_access(self):
        artifact = study.fit(self.v1, self.out)
        self.assertEqual(study.fit(self.v1, self.out), artifact)
        result = study.evaluate(self.v1, self.out)
        before = (self.out / "evaluation_receipts.jsonl").read_bytes()
        self.assertEqual(study.evaluate(self.v1, self.out), result)
        self.assertEqual((self.out / "evaluation_receipts.jsonl").read_bytes(), before)
        self.assertEqual(result["freeze_hash"], artifact["freeze_hash"])
        self.assertEqual(result["primary_difference"]["cluster_count"], 4)
        self.assertEqual(result["evidence_type"], "software_test_fixture")
        # Recomputing a self-hash cannot substitute a different policy after
        # labels have been opened: the independent first-access record binds it.
        artifact["selected_name"] = "changed after access"
        artifact["freeze_hash"] = fingerprint({k: v for k, v in artifact.items() if k != "freeze_hash"})
        write_json(self.out / "policy_freeze.json", artifact)
        with self.assertRaisesRegex(StudyError, "after first holdout access"):
            study.evaluate(self.v1, self.out)

    def test_changed_holdout_predictions_do_not_replace_existing_result(self):
        study.fit(self.v1, self.out)
        before = study.evaluate(self.v1, self.out)
        path = next((self.out / "predictions-jobs/holdout_qwen/predictions/qwen").glob("shard-*.json"))
        shard = read_json(path)
        row = shard["rows"][0]
        row["prediction"] = "unanswerable" if row["prediction"] == "answerable" else "answerable"
        shard["sha256"] = fingerprint(shard["rows"])
        write_json(path, shard)
        with self.assertRaisesRegex(StudyError, "Evaluation identity changed"):
            study.evaluate(self.v1, self.out)
        self.assertEqual(read_json(self.out / "evaluation.json"), before)

    def test_bootstrap_uses_articles_and_question_weighted_denominator(self):
        ex = [{"group_id": "a"}] + [{"group_id": "b"}] * 3
        interval = study._paired_interval(ex, np.array([1, -1, -1, -1]), 2000, 6998)
        self.assertEqual(interval["estimate"], -.5)
        self.assertEqual(interval["cluster_count"], 2)
        self.assertEqual((interval["lower"], interval["upper"]), (-1, 1))


if __name__ == "__main__":
    unittest.main()
