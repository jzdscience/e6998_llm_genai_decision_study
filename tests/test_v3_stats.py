"""Synthetic V3 metric and bootstrap tests; no sealed or downloaded labels."""
import copy
import unittest
from unittest.mock import patch

from decision_study.common import StudyError
from decision_study.v3.stats import cluster_bootstrap_difference, evaluate_direct, evaluate_policy
from test_v3_policies import LABELS, fixture


class V3StatsTests(unittest.TestCase):
    def test_rescues_harms_failures_and_forced_deferral_accounting(self):
        ex, local, fallback, gold = fixture(6, "holdout", [.5,.9,.5,.5,.9,.5])
        for i, x in enumerate(ex):
            sid = x["sample_id"]
            label = local[sid]["prediction"]
            gold[sid] = label if i < 2 else next(y for y in LABELS if y != label)
            fallback[sid]["prediction"] = gold[sid] if i != 0 else next(y for y in LABELS if y != gold[sid])
        local[ex[5]["sample_id"]].update(status="error", prediction=None, probabilities=None)
        fallback[ex[3]["sample_id"]].update(status="error", prediction=None)
        policy = {"family":"confidence", "task":"anli", "labels":list(LABELS), "fallback_model":"qwen9b",
                  "decision":{"mode":"threshold", "threshold":.3, "target_budget":.1}}
        metric, correct = evaluate_policy(policy, ex, local, fallback, gold)
        self.assertEqual(correct, [False,True,True,False,False,True])
        self.assertEqual(metric["correct"], 3)
        self.assertEqual(metric["accuracy"], .5)
        self.assertEqual((metric["deferrals"],metric["forced_deferrals"],metric["optional_deferrals"]), (4,1,3))
        self.assertEqual((metric["rescue"],metric["harm"],metric["net_correct_gain"],metric["final_failures"]), (2,1,1,1))
        direct, bits = evaluate_direct(ex, fallback, gold, model="qwen9b")
        self.assertEqual(direct["correct"], 4)
        self.assertEqual(direct["failures"], 1)
        self.assertEqual(len(bits), 6)
        self.assertNotIn("deferrals", direct)
        with self.assertRaisesRegex(StudyError, "identity"):
            evaluate_direct(ex, fallback, gold, model="qwen4b")

    def test_cluster_bootstrap_is_paired_question_weighted_and_cross_round(self):
        ex, _, _, _ = fixture(6, "holdout")
        # Three unequal premise clusters: A(+3,n3), B(-1,n1), C(0,n2).
        for x, group in zip(ex, ["A","A","A","B","C","C"]):
            x["group_id"] = group
        class KnownDraws:
            def __init__(self):
                self.values = iter([[0,0,0],[1,1,1],[0,1,2],[0,0,2]])
            def integers(self, low, high, size):
                assert (low, high, size) == (0,3,3)
                return next(self.values)
        with patch("decision_study.v3.stats.np.random.default_rng", return_value=KnownDraws()):
            result = cluster_bootstrap_difference(ex, [1,1,1,0,0,0], [0,0,0,1,0,0], repetitions=4)
        # Exact draws are [1,-1,1/3,3/4]; row-weighted estimate = 1/3, not 0.
        self.assertAlmostEqual(result["estimate"], 1/3)
        self.assertAlmostEqual(result["lower"], -.9)
        self.assertAlmostEqual(result["upper"], .98125)
        self.assertEqual(result["cluster_count"], 3)
        self.assertEqual(result["maximum_cluster_size"], 3)
        self.assertEqual(result["cross_round_clusters"], 2)

    def test_default_reproducibility_and_row_permutation_invariance(self):
        ex, _, _, _ = fixture(12, "holdout")
        a, b = [i%2 == 0 for i in range(12)], [i%3 == 0 for i in range(12)]
        first = cluster_bootstrap_difference(ex, a, b)
        self.assertEqual(first, cluster_bootstrap_difference(ex, a, b))
        self.assertEqual(first, cluster_bootstrap_difference(ex[::-1], a[::-1], b[::-1]))
        self.assertEqual((first["replicates"], first["seed"]), (2000,6998))

    def test_degenerate_interval_is_explicit_and_not_a_guarantee(self):
        ex, _, _, _ = fixture(6, "holdout")
        result = cluster_bootstrap_difference(ex, [True]*6, [True]*6, repetitions=20)
        self.assertEqual((result["estimate"],result["lower"],result["upper"]), (0,0,0))
        self.assertEqual(result["warning"], "degenerate_empirical_distribution_not_a_guarantee")

    def test_rejects_misaligned_nonbinary_missing_group_and_single_cluster(self):
        ex, _, _, _ = fixture(6, "holdout")
        for candidate, reference in (([1]*5,[1]*6), ([.5]*6,[1]*6), ([float("nan")]*6,[1]*6)):
            with self.assertRaises(StudyError):
                cluster_bootstrap_difference(ex, candidate, reference)
        for groups in ([None]*6, ["same"]*6):
            altered = copy.deepcopy(ex)
            for x, group in zip(altered, groups):
                x["group_id"] = group
            with self.assertRaises(StudyError):
                cluster_bootstrap_difference(altered, [1]*6, [0]*6)
        with self.assertRaises(StudyError):
            cluster_bootstrap_difference(ex, [1]*6, [0]*6, repetitions=0)


if __name__ == "__main__":
    unittest.main()
