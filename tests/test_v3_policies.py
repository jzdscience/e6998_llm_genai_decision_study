"""Synthetic-only router tests: no dataset loading or model execution."""
import copy
import json
import math
import unittest

from decision_study.common import StudyError
from decision_study.v3.policies import (LABELS, feature_values, fit_logistic, fit_policies,
                                        freeze_budget, route, score_policy)


def fixture(n=30, split="calibration", confidences=None):
    examples, local, fallback, gold = [], {}, {}, {}
    for i in range(n):
        sid = f"synthetic-{split}-{i}"
        label = LABELS[i % 3]
        c = confidences[i] if confidences is not None else .5 + (i % 10) * .04
        example = {"sample_id": sid, "task": "anli", "split": split, "labels": list(LABELS),
                   "state": "synthetic" * (i + 1), "group_id": f"synthetic-premise-{i // 3}",
                   "round": f"r{1 + i % 3}"}
        examples.append(example)
        gold[sid] = label if i % 4 else LABELS[(i + 1) % 3]
        local[sid] = {"sample_id": sid, "model": "startlux4b", "status": "ok", "prediction": label,
                      "probabilities": {x: c if x == label else (1-c)/2 for x in LABELS}}
        fallback[sid] = {"sample_id": sid, "model": "qwen9b", "status": "ok", "prediction": gold[sid],
                         "probabilities": None}
    return examples, local, fallback, gold


def gain_policy(family="positive_gain"):
    """Known score = confidence - .5, using all five frozen features."""
    return {"family": family, "task": "anli", "labels": list(LABELS), "include_length": True,
            "fallback_model": "qwen9b", "local_model": "startlux4b",
            "local_head": {"kind": "constant", "probability": .5},
            "fallback_head": {"kind": "logistic", "feature_mean": [0]*5, "feature_scale": [1]*5,
                              "coefficients": [0, 1, 0, 0, 0, 0]}}


class V3PolicyTests(unittest.TestCase):
    def test_three_class_features_use_all_one_hot_columns_and_public_length(self):
        ex, local, _, _ = fixture(3)
        for i, x in enumerate(ex):
            features = feature_values(x, local[x["sample_id"]])
            self.assertEqual(len(features), 5)
            self.assertEqual(features[1:4], [float(j == i) for j in range(3)])
            self.assertAlmostEqual(features[0], math.log((.5 + i*.04)/(1-.5-i*.04)))
            self.assertEqual(features[4], math.log1p(len(x["state"])))

    def test_heads_fit_observed_correctness_even_if_local_confident(self):
        ex, local, fallback, gold = fixture(18)
        for x in ex:
            sid = x["sample_id"]
            row = local[sid]
            gold[sid] = next(label for label in LABELS if label != row["prediction"])
            row["probabilities"] = {label: .98 if label == row["prediction"] else .01 for label in LABELS}
            fallback[sid]["prediction"] = gold[sid]
        fitted = fit_policies(ex, local, fallback, gold, fallback_model="qwen9b")
        self.assertEqual(set(fitted), {"confidence", "class_conditional", "fixed_gain", "positive_gain"})
        self.assertEqual(fitted["positive_gain"]["local_head"]["positives"], 0)
        self.assertEqual(fitted["positive_gain"]["fallback_head"]["positives"], 18)
        self.assertGreater(score_policy(fitted["positive_gain"], ex[0], local[ex[0]["sample_id"]]), .8)
        json.dumps(fitted, allow_nan=False)

    def test_negative_gain_can_reject_every_optional_request(self):
        ex, local, fallback, gold = fixture(12)
        for x in ex:
            sid = x["sample_id"]
            gold[sid] = local[sid]["prediction"]
            fallback[sid]["prediction"] = next(y for y in LABELS if y != gold[sid])
        fitted = fit_policies(ex, local, fallback, gold, fallback_model="qwen9b")
        val, rows, _, _ = fixture(10, "validation")
        positive = freeze_budget(fitted["positive_gain"], val, rows, 1)
        fixed = freeze_budget(fitted["fixed_gain"], val, rows, 1)
        self.assertEqual(positive["decision"]["selection_optional_deferrals"], 0)
        self.assertEqual(positive["decision"]["mode"], "none")
        self.assertTrue(all(not route(positive, x, rows[x["sample_id"]]) for x in val))
        self.assertTrue(all(route(fixed, x, rows[x["sample_id"]]) for x in val))

    def test_zero_gain_excluded_during_freeze_independently_of_route(self):
        ex, local, _, _ = fixture(3, "validation", [.5]*3)
        frozen = freeze_budget(gain_policy(), ex, local, 1)
        self.assertEqual(frozen["decision"]["eligible_positive_or_unrestricted"], 0)
        self.assertEqual(frozen["decision"]["selection_optional_deferrals"], 0)

    def test_positive_guard_at_runtime_even_with_negative_frozen_threshold(self):
        ex, local, _, _ = fixture(3, "holdout", [.4, .5, .6])
        policy = gain_policy()
        policy["decision"] = {"mode": "threshold", "threshold": -.2}
        self.assertEqual([route(policy, x, local[x["sample_id"]]) for x in ex], [False, False, True])

    def test_entire_tie_groups_and_exact_threshold_equality(self):
        ex, local, _, _ = fixture(10, "validation", [.9]*2 + [.8]*3 + [.6]*5)
        frozen = freeze_budget(gain_policy(), ex, local, .2)
        self.assertEqual(frozen["decision"]["selection_deferrals"], 2)
        self.assertEqual([route(frozen, x, local[x["sample_id"]]) for x in ex], [True]*2 + [False]*8)
        self.assertEqual(frozen["decision"]["threshold"], score_policy(frozen, ex[0], local[ex[0]["sample_id"]]))

    def test_oversized_highest_tie_is_not_split_or_skipped_for_lower_group(self):
        ex, local, _, _ = fixture(10, "validation", [.9]*3 + [.8] + [.6]*6)
        frozen = freeze_budget(gain_policy(), ex, local, .2)
        self.assertEqual(frozen["decision"]["selection_deferrals"], 0)
        self.assertEqual(frozen["decision"]["mode"], "none")

    def test_failure_overrides_defer_none_and_can_exceed_budget(self):
        ex, local, _, _ = fixture(10, "validation", [.4]*10)
        local[ex[0]["sample_id"]].update(status="error", prediction=None, probabilities=None)
        frozen = freeze_budget(gain_policy(), ex, local, 0)
        self.assertEqual(frozen["decision"]["mandatory_failures"], 1)
        self.assertTrue(frozen["decision"]["budget_exceeded_by_failures"])
        self.assertEqual([route(frozen, x, local[x["sample_id"]]) for x in ex], [True]+[False]*9)

    def test_failure_consumes_optional_budget_first(self):
        ex, local, _, _ = fixture(10, "validation", [.9, .85, .8, .75, .7, .65, .6, .55, .5, .4])
        local[ex[-1]["sample_id"]].update(status="error", prediction=None, probabilities=None)
        frozen = freeze_budget(gain_policy(), ex, local, .2)
        self.assertEqual(frozen["decision"]["selection_optional_deferrals"], 1)
        self.assertEqual(sum(route(frozen, x, local[x["sample_id"]]) for x in ex), 2)

    def test_holdout_budget_shift_keeps_threshold_and_never_rebalances(self):
        ex, local, _, _ = fixture(10, "validation", [.9]+[.6]*9)
        frozen = freeze_budget(gain_policy(), ex, local, .1)
        holdout, rows, _, _ = fixture(20, "holdout", [.95]*20)
        self.assertEqual(sum(route(frozen, x, rows[x["sample_id"]]) for x in holdout), 20)
        self.assertEqual(frozen["decision"]["selection_deferrals"], 1)

    def test_selecting_all_validation_candidates_still_keeps_a_threshold(self):
        ex, local, _, _ = fixture(4, "validation", [.8]*4)
        frozen = freeze_budget(gain_policy(), ex, local, 1)
        holdout, rows, _, _ = fixture(1, "holdout", [.6])
        self.assertEqual(frozen["decision"]["mode"], "threshold")
        self.assertFalse(route(frozen, holdout[0], rows[holdout[0]["sample_id"]]))

    def test_calibration_only_scaling_and_copy_isolation(self):
        ex, local, fallback, gold = fixture(12)
        val, rows, _, _ = fixture(10, "validation")
        base = fit_policies(ex, local, fallback, gold, fallback_model="qwen9b")["positive_gain"]
        expected_length_mean = sum(math.log1p(len(x["state"])) for x in ex)/len(ex)
        self.assertAlmostEqual(base["local_head"]["feature_mean"][-1], expected_length_mean)
        before = copy.deepcopy(base)
        for x in val:
            x["state"] = "x"*10000
        frozen = freeze_budget(base, val, rows)
        self.assertEqual(base, before)
        self.assertEqual(frozen["local_head"], base["local_head"])
        frozen["local_head"]["feature_mean"][0] = 999
        self.assertEqual(base, before)

    def test_role_and_identity_guards(self):
        ex, local, fallback, gold = fixture(6)
        with self.assertRaisesRegex(StudyError, "identity"):
            fit_policies(ex, local, fallback, gold, fallback_model="qwen4b")
        fitted = fit_policies(ex, local, fallback, gold, fallback_model="qwen9b")
        with self.assertRaisesRegex(StudyError, "validation"):
            freeze_budget(fitted["confidence"], ex, local)
        val, rows, backs, labels = fixture(6, "holdout")
        with self.assertRaisesRegex(StudyError, "calibration"):
            fit_policies(val, rows, backs, labels, fallback_model="qwen9b")
        with self.assertRaises(StudyError):
            fit_policies(ex+[ex[0]], local, fallback, gold, fallback_model="qwen9b")
        with self.assertRaises(StudyError):
            score_policy(fitted["confidence"], ex[0], local[ex[1]["sample_id"]])
        with self.assertRaisesRegex(StudyError, "frozen"):
            route(fitted["confidence"], ex[0], local[ex[0]["sample_id"]])

    def test_invalid_multiclass_probability_vector_is_not_silently_scored(self):
        ex, local, _, _ = fixture(1)
        row = local[ex[0]["sample_id"]]
        for probs in ({"entailment": 1}, dict(zip(LABELS, [.2,.4,.4])),
                      dict(zip(LABELS, [.8,.1,float("nan")])), dict(zip(LABELS, [.8,.2,.2]))):
            bad = {**row, "probabilities": probs}
            with self.assertRaises(StudyError):
                feature_values(ex[0], bad)

    def test_failed_local_excluded_but_failed_fallback_is_incorrect_target(self):
        ex, local, fallback, gold = fixture(9)
        local[ex[0]["sample_id"]].update(status="error", prediction=None, probabilities=None)
        for row in fallback.values():
            row.update(status="error", prediction=None)
        base = fit_policies(ex, local, fallback, gold, fallback_model="qwen9b")["positive_gain"]
        self.assertEqual(base["usable_fit_count"], 8)
        self.assertEqual(base["local_failures_excluded_from_features"], 1)
        self.assertEqual(base["fallback_head"]["positives"], 0)

    def test_router_ignores_gold_and_fallback_fields_on_public_example(self):
        ex, local, _, _ = fixture(1)
        row = local[ex[0]["sample_id"]]
        value = score_policy(gain_policy(), ex[0], row)
        extra = {**ex[0], "gold": "contradiction", "fallback_prediction": "entailment", "fallback_correct": True}
        self.assertEqual(score_policy(gain_policy(), extra, row), value)

    def test_logistic_learns_signal_and_constant_smoothing(self):
        head = fit_logistic([[-2],[-1],[1],[2]]*10, [0,0,1,1]*10)
        self.assertGreater(head["coefficients"][1], 0)
        self.assertLess(head["objective"], .4)
        self.assertEqual(fit_logistic([[0],[1]], [1,1])["probability"], .75)
        with self.assertRaises(StudyError):
            fit_logistic([[0],[1]], [0,1], l2=0)


if __name__ == "__main__":
    unittest.main()
