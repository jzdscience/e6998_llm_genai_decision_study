import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from decision_study.analysis import (apply_gate, classification_metrics, evaluate_experiment,
    fit_model, fit_temperature, paired_bootstrap, probability_metrics, rescale_probabilities,
    risk_coverage_curve, routing_controls, routing_metrics, routing_reference_curve, select_gate, validate_calibration)


class AnalysisTests(unittest.TestCase):
    def test_temperature_improves_nll_and_preserves_binary_ranking(self):
        p = np.array([[.99, .01]] * 20 + [[.01, .99]] * 20)
        y = np.array([0] * 15 + [1] * 5 + [1] * 15 + [0] * 5)
        fitted = fit_temperature(p, y)
        self.assertGreater(fitted["temperature"], 1)
        self.assertLess(fitted["nll_after"], fitted["nll_before"])
        heterogeneous = np.array([[.95, .05], [.2, .8], [.51, .49], [.3, .7]])
        for t in (.05, .8, 3, 20):
            scaled = rescale_probabilities(heterogeneous, t)
            np.testing.assert_array_equal(heterogeneous.argmax(1), scaled.argmax(1))
            np.testing.assert_array_equal(np.argsort(heterogeneous.max(1)), np.argsort(scaled.max(1)))

    def test_gate_never_splits_ties(self):
        gate = select_gate([.9, .9, .8], [True, False, True], min_accepted=1, max_risk=.05)
        self.assertTrue(gate["accept_none"])
        curve = risk_coverage_curve(np.array([.9, .9, .8]), np.array([1, 0, 1], bool), np.ones(3, bool))
        self.assertEqual([p["accepted"] for p in curve], [0, 2, 3])

    def test_gate_can_become_feasible_after_an_infeasible_prefix(self):
        confidence = np.arange(100, 0, -1) / 100
        correct = np.ones(100, dtype=bool)
        correct[0] = False
        gate = select_gate(confidence, correct)
        self.assertEqual(gate["validation_accepted"], 100)
        self.assertEqual(gate["threshold"], .01)

    def test_invalid_rows_are_forced_fallback_and_zero_acceptance_is_na(self):
        accepted = apply_gate([.99, .99], [True, False], {"accept_none": False, "threshold": .9})
        np.testing.assert_array_equal(accepted, [True, False])
        summary, correct = routing_metrics([True, False], [False, False], accepted)
        self.assertEqual(summary["cascade_accuracy"], .5)
        summary, _ = routing_metrics([False, False], [True, False], np.zeros(2, bool))
        self.assertIsNone(summary["selective_risk"])
        self.assertIsNone(summary["target_miss"])

    def test_oracle_exact_quota_can_force_negative_gain(self):
        c = routing_controls([True, True], [False, True], [False, False], 2, random_reps=10)
        self.assertEqual(c["oracle"]["accuracy"], .5)
        self.assertEqual(c["oracle"]["negative_gain_deferrals"], 1)
        with self.assertRaises(ValueError):
            routing_controls([False, True], [True, False], [True, False], 0)

    def test_oracle_obeys_mandatory_failures(self):
        c = routing_controls([False, True, False], [False, False, True], [True, False, False], 2, random_reps=10)
        self.assertEqual(c["mandatory_deferrals"], 1)
        self.assertEqual(c["oracle"]["accuracy"], 2 / 3)
        self.assertAlmostEqual(c["random"]["expected_accuracy"], 1 / 3)

    def test_oracle_reference_includes_quotas_between_confidence_ties(self):
        curve = routing_reference_curve([False, True], [True, False], [True, True])
        self.assertEqual([p["deferral_count"] for p in curve], [0, 1, 2])
        self.assertEqual(curve[1]["oracle_accuracy"], 1)
        self.assertEqual(curve[1]["random_expected_accuracy"], .5)

    def test_probability_metrics_and_failure_denominator(self):
        m = probability_metrics(np.array([[.8, .2], [.2, .8]]), np.array([0, 1]))
        self.assertAlmostEqual(m["brier"], .08)
        self.assertAlmostEqual(m["ece"], .2)
        c = classification_metrics(np.array([0, 1]), np.array([0, -1]), ["a", "b"])
        self.assertEqual(c["accuracy"], .5)
        self.assertEqual(c["failure_rate"], .5)
        self.assertEqual(c["confusion"]["counts"], [[1, 0, 0], [0, 0, 1]])

    def test_cluster_bootstrap_pairs_systems_and_flags_zero_errors(self):
        ex = [{"sample_id": str(i), "task": "squad", "group_id": str(i // 2)} for i in range(8)]
        gold = {str(i): "a" for i in range(8)}
        correct = np.array([1, 1, 0, 0, 1, 1, 0, 0], bool)
        conditions = {"direct/qwen": {"correct": correct}, "direct/kev": {"correct": correct.copy()},
                      "cascade/kev/refitted": {"correct": correct, "accepted": correct.copy(), "local_correct": correct}}
        result = paired_bootstrap(ex, gold, conditions, repetitions=100)
        self.assertEqual(result["cluster_count"], 4)
        self.assertEqual(result["paired_accuracy_differences"]["direct/kev - direct/qwen"]["lower"], 0)
        self.assertIsNone(result["intervals"]["cascade/kev/refitted"]["selective_risk"]["upper"])

    @staticmethod
    def fixture():
        # Deliberately small artificial fixture; these are software tests, not research results.
        examples, gold, rows = [], {}, {m: [] for m in ("kev", "startlux", "qwen")}
        for task, labels in (("clinc", ["bank", "oos"]), ("squad", ["answerable", "unanswerable"])):
            for split in ("calibration", "validation", "test", "stress"):
                if task == "squad" and split == "stress":
                    continue
                n = 8 if split != "stress" else 4
                for i in range(n):
                    sid = f"{task}-{split}-{i}"
                    y = 1 if split == "stress" else i % 2
                    ex = {"sample_id": sid, "task": task, "split": split, "group_id": f"{task}-{split}-article{i // 2}",
                          "labels": labels, "criteria": {k: k for k in labels}, "state": "Artificial test fixture", "instructions": "Choose"}
                    examples.append(ex)
                    gold[sid] = labels[y]
                    for m in rows:
                        if m == "qwen" and split not in ("test", "stress"):
                            continue
                        guess = y if i % (3 if m == "qwen" else 4) else 1 - y
                        rows[m].append({"sample_id": sid, "model": m, "config_hash": f"frozen-{m}", "status": "ok",
                                        "prediction": labels[guess], "probabilities": None if m == "qwen" else {labels[guess]: .8, labels[1 - guess]: .2},
                                        "attempts": [{}], "is_test_fixture": True})
        return examples, gold, rows

    def test_end_to_end_fit_evaluate_and_calibration_provenance(self):
        ex, gold, rows = self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            cal = {m: fit_model(ex, gold, rows[m], m, tmp) for m in ("kev", "startlux")}
            result = evaluate_experiment(ex, gold, rows, cal, tmp, bootstrap_reps=30, random_reps=10)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["datasets"]["clinc"]["main"]["n"], 8)
            self.assertEqual(result["datasets"]["clinc"]["stress"]["n"], 8)
            self.assertTrue((Path(tmp) / "analysis" / "summary.csv").is_file())
            saved = json.loads((Path(tmp) / "analysis" / "metrics.json").read_text())
            self.assertEqual(saved["config_hashes"]["kev"], "frozen-kev")
            stale = copy.deepcopy(cal)
            stale["kev"]["analysis_source_hash"] = "old-implementation"
            with self.assertRaisesRegex(ValueError, "analysis source mismatch"):
                evaluate_experiment(ex, gold, rows, stale, tmp, 2, 2)
            changed = copy.deepcopy(rows)
            changed["kev"][0]["prediction"] = "oos"
            changed["kev"][0]["probabilities"] = {"bank": .1, "oos": .9}
            with self.assertRaisesRegex(ValueError, "prediction mismatch"):
                evaluate_experiment(ex, gold, changed, cal, tmp, 2, 2)
            missing = copy.deepcopy(rows)
            missing["qwen"].pop()
            with self.assertRaisesRegex(ValueError, "Incomplete predictions"):
                evaluate_experiment(ex, gold, missing, cal, tmp, 2, 2)

    def test_calibration_checksum_rejects_threshold_tampering_and_unknown_ids(self):
        ex, gold, rows = self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            artifact = fit_model(ex, gold, rows["kev"], "kev", tmp)
            self.assertIsNone(validate_calibration(ex, gold, rows["kev"], artifact, "kev"))
            tampered = copy.deepcopy(artifact)
            tampered["tasks"]["clinc"]["gates"]["refitted"]["threshold"] = .99
            with self.assertRaisesRegex(ValueError, "artifact checksum mismatch"):
                validate_calibration(ex, gold, rows["kev"], tampered, "kev")
            extra = copy.deepcopy(rows["kev"])
            extra.append({**extra[0], "sample_id": "foreign-dataset-input"})
            with self.assertRaisesRegex(ValueError, "Unexpected prediction IDs"):
                validate_calibration(ex, gold, extra, artifact, "kev")
            with self.assertRaisesRegex(ValueError, "Unexpected prediction IDs"):
                fit_model(ex, gold, extra, "kev", tmp)

    def test_mixed_configuration_rejected(self):
        ex, gold, rows = self.fixture()
        rows["kev"][0]["config_hash"] = "different"
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "config_hash"):
                fit_model(ex, gold, rows["kev"], "kev", tmp)


if __name__ == "__main__":
    unittest.main()
