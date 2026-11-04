import copy
import unittest

from decision_study.common import StudyError
from decision_study.v2.policies import (fit_candidates, fit_logistic, freeze_budget,
                                       score_policy, should_defer)


def fixture(n=20, split="calibration"):
    examples, gold, local, qwen = [], {}, {}, {}
    labels = ["answerable", "unanswerable"]
    for i in range(n):
        sid = f"fixture-{split}-{i}"
        predicted = labels[i % 2]
        example = {"sample_id": sid, "task": "squad", "split": split,
                   "labels": labels, "state": "x" * (10 + 11 * i), "group_id": f"{split}-{i // 4}"}
        examples.append(example)
        gold[sid] = predicted if i % 4 else labels[1 - i % 2]
        c = .55 + .02 * (i % 20)
        local[sid] = {"sample_id": sid, "model": "startlux", "status": "ok", "prediction": predicted,
                      "probabilities": {predicted: c, labels[1 - i % 2]: 1 - c}, "is_test_fixture": True}
        qwen[sid] = {"sample_id": sid, "model": "qwen", "status": "ok", "prediction": gold[sid],
                     "probabilities": None, "is_test_fixture": True}
    return examples, gold, local, qwen


class PolicyTests(unittest.TestCase):
    def test_two_heads_use_observed_correctness_not_probability_shortcut(self):
        ex, gold, local, qwen = fixture(12)
        for x in ex:
            r = local[x["sample_id"]]
            other = next(label for label in x["labels"] if label != r["prediction"])
            r["probabilities"] = {r["prediction"]: .99, other: .01}
            gold[x["sample_id"]] = other
            qwen[x["sample_id"]]["prediction"] = other
        fitted = fit_candidates(ex, gold, local, qwen)
        self.assertEqual(fitted["gain"]["local_head"]["positives"], 0)
        self.assertEqual(fitted["gain"]["qwen_head"]["positives"], 12)
        self.assertGreater(score_policy(fitted["gain"], ex[0], local[ex[0]["sample_id"]]), .8)

    def test_fitting_and_threshold_roles_are_enforced(self):
        ex, gold, local, qwen = fixture(split="validation")
        with self.assertRaisesRegex(StudyError, "calibration"):
            fit_candidates(ex, gold, local, qwen)
        ex, gold, local, qwen = fixture()
        base = fit_candidates(ex, gold, local, qwen)["confidence"]
        with self.assertRaisesRegex(StudyError, "validation"):
            freeze_budget(base, ex, local, .1)

    def test_threshold_never_splits_ties_and_failures_are_mandatory(self):
        ex, _, local, _ = fixture(4, "validation")
        for x in ex:
            label = local[x["sample_id"]]["prediction"]
            local[x["sample_id"]]["probabilities"] = {label: .8, next(k for k in x["labels"] if k != label): .2}
        base = {"family": "confidence", "task": "squad", "labels": ex[0]["labels"]}
        policy = freeze_budget(base, ex, local, .25)
        self.assertFalse(any(should_defer(policy, x, local[x["sample_id"]]) for x in ex))
        local[ex[0]["sample_id"]].update(status="error", probabilities=None, prediction=None)
        policy = freeze_budget(base, ex, local, 0)
        self.assertTrue(policy["decision"]["budget_exceeded_by_failures"])
        self.assertEqual(sum(should_defer(policy, x, local[x["sample_id"]]) for x in ex), 1)

    def test_runtime_policy_does_not_use_gold_or_qwen_features(self):
        ex, gold, local, qwen = fixture()
        fitted = fit_candidates(ex, gold, local, qwen)["gain"]
        a = score_policy(fitted, ex[0], local[ex[0]["sample_id"]])
        altered = {**ex[0], "gold": "unanswerable", "qwen_correct": False, "qwen_prediction": "unanswerable"}
        b = score_policy(fitted, altered, local[ex[0]["sample_id"]])
        self.assertEqual(a, b)

    def test_labels_and_ids_cannot_silently_misbind(self):
        ex, _, local, _ = fixture(4, "validation")
        p = {"family": "confidence", "task": "squad", "labels": ex[0]["labels"]}
        bad = copy.deepcopy(local[ex[0]["sample_id"]])
        bad["probabilities"] = {"not_the_task": 1.0}
        with self.assertRaises(StudyError):
            score_policy(p, ex[0], bad)
        with self.assertRaises(StudyError):
            score_policy(p, ex[0], local[ex[1]["sample_id"]])

    def test_original_gate_retains_accept_none_and_equality_semantics(self):
        ex, _, local, _ = fixture(1, "validation")
        p = {"family": "original", "task": "squad", "labels": ex[0]["labels"], "temperature": 1,
             "gate": {"threshold": .55, "accept_none": False}}
        self.assertFalse(should_defer(p, ex[0], local[ex[0]["sample_id"]]))
        p["gate"]["accept_none"] = True
        self.assertTrue(should_defer(p, ex[0], local[ex[0]["sample_id"]]))

    def test_logistic_learns_signal_and_handles_constant_target(self):
        model = fit_logistic([[-2], [-1], [1], [2]] * 10, [0, 0, 1, 1] * 10)
        self.assertGreater(model["coefficients"][1], 0)
        self.assertLess(model["objective"], .4)
        self.assertEqual(fit_logistic([[0], [1]], [1, 1])["probability"], .75)


if __name__ == "__main__":
    unittest.main()
