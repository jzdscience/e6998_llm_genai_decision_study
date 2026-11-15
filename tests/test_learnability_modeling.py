"""Small, explicitly artificial routing/solver fixtures; no real holdout scores."""
import copy
import json
import math

import numpy as np
import pytest

from decision_study.common import StudyError, fingerprint
from decision_study.learnability import modeling as m


def test_logistic_objective_gradient_matches_finite_differences():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(17, 5)); y = rng.integers(2, size=17).astype(float)
    beta = rng.normal(size=6)
    value, gradient = m.logistic_objective_gradient(beta, x, y, .1)
    finite = []
    for j in range(len(beta)):
        eps = np.zeros_like(beta); eps[j] = 1e-6
        hi = m.logistic_objective_gradient(beta+eps, x, y, .1)[0]
        lo = m.logistic_objective_gradient(beta-eps, x, y, .1)[0]
        finite.append((hi-lo)/2e-6)
    np.testing.assert_allclose(gradient, finite, atol=1e-9, rtol=1e-8)
    unpenalized = m.logistic_objective_gradient(beta, x, y, 0)[1]
    assert gradient[0] == unpenalized[0]
    np.testing.assert_allclose(gradient[1:]-unpenalized[1:], .1*beta[1:])
    assert math.isfinite(value)


def test_solver_improves_objective_converges_and_serialized_scores_replay():
    x = np.column_stack([np.linspace(-3, 3, 80), np.ones(80)])
    y = (x[:, 0] > .2).astype(float)
    head = m.fit_head(x, y, .1)
    assert head["kind"] == "logistic" and head["gradient_inf_norm"] <= 1e-6
    z = (x-np.asarray(head["mean"]))/np.asarray(head["scale"])
    beta = np.asarray(head["coefficients"])
    value, gradient = m.logistic_objective_gradient(beta, z, y, .1)
    initial = np.zeros_like(beta); initial[0] = math.log(y.mean()/(1-y.mean()))
    assert value < m.logistic_objective_gradient(initial, z, y, .1)[0]
    assert np.max(np.abs(gradient)) <= 1e-6
    assert head["scale"][1] == 1
    probability = m.score_head(json.loads(json.dumps(head)), x)
    assert np.all(np.diff(probability) > 0)
    np.testing.assert_allclose(probability, 1/(1+np.exp(-(beta[0]+z@beta[1:]))), atol=1e-14)


@pytest.mark.parametrize("target,expected", [(0, .5/11), (1, 10.5/11)])
def test_constant_target_uses_fixed_jeffreys_recipe(target, expected):
    head = m.fit_head(np.ones((10, 3)), np.full(10, target), .1)
    assert head["kind"] == "constant" and head["iterations"] == 0
    np.testing.assert_array_equal(m.score_head(head, np.zeros((4, 3))), np.full(4, expected))


def test_solver_failure_is_not_silently_accepted(monkeypatch):
    monkeypatch.setitem(m.RECIPE, "max_iterations", 0)
    with pytest.raises(StudyError, match="iteration budget"):
        m.fit_head(np.arange(10)[:, None], np.asarray([0]*5+[1]*5), .1)


def test_exact_quota_uses_stable_ids_and_includes_forced_failures():
    ids = [f"id{i:02}" for i in range(20)][::-1]
    mask = m.exact_quota(np.zeros(20), ids)
    assert set(np.asarray(ids)[mask]) == {"id00", "id01"}
    forced = np.zeros(20, dtype=bool); forced[0] = True
    mask = m.exact_quota(np.zeros(20), ids, forced)
    assert set(np.asarray(ids)[mask]) == {"id19", "id00"}
    forced[:3] = True
    with pytest.raises(StudyError, match="exceed"):
        m.exact_quota(np.zeros(20), ids, forced)


def test_exact_quota_floor_zero_and_bootstrap_duplicate_ties():
    assert m.exact_quota(np.ones(9), [str(i) for i in range(9)]).sum() == 0
    mask = m.exact_quota(np.ones(20), ["same"]*20)
    assert np.flatnonzero(mask).tolist() == [0, 1]
    with pytest.raises(StudyError):
        m.exact_quota([float("nan")]*10, list(map(str, range(10))))


def test_threshold_allows_none_and_ties_prefer_fewer_deferrals():
    scores = np.linspace(.1, 1, 20)
    ids = list(map(str, range(20)))
    front = np.ones(20, dtype=bool); back = front.copy(); forced = np.zeros(20, dtype=bool)
    assert m._threshold(scores, ids, front, back, forced)["threshold"] is None
    front[-1] = False
    value = m._threshold(scores, ids, front, back, forced)
    assert value == {"threshold": 1.0, "net_correct": 1, "deferred": 1}
    back[-2] = False
    assert m._threshold(scores, ids, front, back, forced) == value


@pytest.fixture
def artificial(monkeypatch):
    monkeypatch.setattr(m, "FIT_SIZES", (18, 36))
    monkeypatch.setattr(m, "EMBEDDING_DIM", 6)
    monkeypatch.setattr(m, "BOOTSTRAP_REPLICATES", 80)
    recipe = copy.deepcopy(m.RECIPE); recipe.update(text_dimension=6, bootstrap_replicates=80)
    monkeypatch.setattr(m, "RECIPE", recipe)
    rng = np.random.default_rng(31)
    examples, front, backs, gold, embeddings = [], {}, {r: {} for r in m.FALLBACKS}, {}, {}
    for split, count in [("fitting", 36), ("selection", 18), ("eligible_dev", 30), ("train_holdout", 30)]:
        for i in range(count):
            sid = f"artificial:{split}:{i:03}"
            ex = {"sample_id": sid, "task": "anli", "split": split, "labels": list(m.LABELS),
                  "state": "Artificial premise and hypothesis " + "x"*(i+1), "round": f"R{i%3+1}",
                  "premise_hash": f"{split}-g{i//2}"}
            examples.append(ex); gold[sid] = m.LABELS[i%3]
            def row(correct, probabilities=False):
                label = gold[sid] if correct else m.LABELS[(i+1)%3]
                value = {"sample_id": sid, "status": "ok", "prediction": label}
                if probabilities:
                    value["probabilities"] = {k: .75 if k == label else .125 for k in m.LABELS}
                return value
            front[sid] = row(i%4 != 0, True)
            backs["qwen9b"][sid] = row(i%3 != 0)
            backs["startlux9b"][sid] = row(i%5 != 0)
            vector = rng.normal(size=6); embeddings[sid] = (vector/np.linalg.norm(vector)).tolist()
    by_split = {split: [e for e in examples if e["split"] == split] for split in {e["split"] for e in examples}}
    fit_ids = [e["sample_id"] for e in by_split["fitting"]]
    train_ids = fit_ids+[e["sample_id"] for e in by_split["selection"]]
    train_gold = {sid: gold[sid] for sid in train_ids}
    return {"splits": by_split, "front": front, "backs": backs, "gold": gold, "train_gold": train_gold,
            "embedding": embeddings, "nested": {"18": fit_ids[:18], "36": fit_ids}}


def fit(a):
    return m.fit_suite(a["splits"]["fitting"], a["splits"]["selection"], a["front"], a["backs"],
                       a["train_gold"], a["embedding"], a["nested"])


def test_features_are_exact_front_fields_and_normalized_text_only(artificial):
    a = artificial; ex = a["splits"]["fitting"][:1]; row = a["front"][ex[0]["sample_id"]]
    matrices, failed, confidence = m.feature_matrices(ex, a["front"], a["embedding"])
    assert matrices["front"].shape == (1, 7) and matrices["text"].shape == (1, 13)
    assert confidence.tolist() == [.75] and failed.tolist() == [False]
    np.testing.assert_allclose(matrices["front"][0, :3], [row["probabilities"][k] for k in m.LABELS])
    assert matrices["front"][0, 3+m.LABELS.index(row["prediction"])] == 1
    assert matrices["front"][0, 6] == math.log1p(len(ex[0]["state"]))
    a["embedding"][ex[0]["sample_id"]] = [2.0]*6
    with pytest.raises(StudyError, match="normalized"):
        m.feature_matrices(ex, a["front"], a["embedding"])


@pytest.mark.parametrize("mutation", ["extra_eval_gold", "eval_in_fit", "same_premise", "not_nested", "gold_in_input"])
def test_fitting_rejects_role_leakage_and_bad_nested_ids(artificial, mutation):
    a = artificial
    if mutation == "extra_eval_gold":
        sid = a["splits"]["eligible_dev"][0]["sample_id"]; a["train_gold"][sid] = a["gold"][sid]
    elif mutation == "eval_in_fit":
        a["splits"]["fitting"][0]["split"] = "eligible_dev"
    elif mutation == "same_premise":
        a["splits"]["selection"][0]["premise_hash"] = a["splits"]["fitting"][0]["premise_hash"]
    elif mutation == "not_nested":
        a["nested"]["18"][0] = "unknown"
    else:
        a["splits"]["fitting"][0]["gold"] = "entailment"
    with pytest.raises(StudyError):
        fit(a)


def test_end_to_end_artificial_suite_freezes_without_eval_gold_then_scores(artificial):
    a = artificial; models = fit(a)
    assert models["evaluation_gold_used"] is False and len(models["cells"]) == 8
    expected_sizes = {18, 36}
    for cell in models["cells"].values():
        assert cell["fit_size"] in expected_sizes
        assert [c["l2"] for c in cell["candidates"]] == [1, .1, .01]
        choice = max(range(3), key=lambda j: (cell["candidates"][j]["selection"]["correct"], cell["candidates"][j]["l2"]))
        assert cell["selected_index"] == choice
        for candidate in cell["candidates"]:
            assert candidate["local_head"]["l2"] == candidate["fallback_head"]["l2"]
    for split in ("eligible_dev", "train_holdout"):
        rows = a["splits"][split]
        frozen = m.score_suite(models, rows, a["front"], a["embedding"])
        assert frozen["evaluation_gold_used"] is False and frozen["models_sha256"] == fingerprint(models)
        result = m.evaluate_suite(frozen, rows, a["front"], a["backs"], {e["sample_id"]: a["gold"][e["sample_id"]] for e in rows})
        assert all(c["deferred"] == 3 for c in result["cells"].values())
        primary = result["comparisons"]["qwen9b/n36"]
        assert primary["primary"] == (split == "eligible_dev")
        assert primary["interval"]["replicates"] == 80
        assert primary["interval"]["premise_groups"] == 15
        assert len(primary["interval"]["ci95"]) == 2
        for name, cell in result["cells"].items():
            role = name.split("/")[0]
            assert result["baselines"][f"{role}/oracle10_hindsight"]["correct"] >= cell["correct"]
            assert cell["correct"] == result["baselines"]["direct_startlux4b"]["correct"] + cell["rescues"]-cell["harms"]


def test_standardization_uses_fitting_only_and_score_does_not_mutate_model(artificial):
    a = artificial; models = fit(a)
    x, _, _ = m.feature_matrices(a["splits"]["fitting"][:18], a["front"], a["embedding"])
    head = models["cells"]["qwen9b/n18/text"]["candidates"][0]["local_head"]
    np.testing.assert_allclose(head["mean"], x["text"].mean(0))
    frozen_models = json.dumps(models, sort_keys=True)
    rows = a["splits"]["eligible_dev"]
    for row in rows: row["state"] += "y"*10000
    m.score_suite(models, rows, a["front"], a["embedding"])
    assert json.dumps(models, sort_keys=True) == frozen_models


def test_selection_tamper_or_changed_front_observation_rejected(artificial):
    a = artificial; models = fit(a); rows = a["splits"]["eligible_dev"]
    frozen = m.score_suite(models, rows, a["front"], a["embedding"])
    bad = copy.deepcopy(models); key = next(iter(bad["cells"]))
    bad["cells"][key]["selected_index"] = (bad["cells"][key]["selected_index"]+1)%3
    with pytest.raises(StudyError, match="selection/tie"):
        m.score_suite(bad, rows, a["front"], a["embedding"])
    sid = rows[0]["sample_id"]; row = a["front"][sid]
    row["probabilities"] = {k: .6 if k == row["prediction"] else .2 for k in m.LABELS}
    with pytest.raises(StudyError, match="front observation"):
        m.evaluate_suite(frozen, rows, a["front"], a["backs"], {e["sample_id"]: a["gold"][e["sample_id"]] for e in rows})


def test_bootstrap_recomputes_quota_with_variable_cluster_sizes(monkeypatch):
    # One 1-row group and one 19-row group. A resample of a single small group
    # twice has N=2,k=0, and of the large group twice N=38,k=3; never reuse k=2.
    rows = [{"sample_id": f"i{i}", "premise_hash": "small" if i == 0 else "large"} for i in range(20)]
    observed = []
    original = m.exact_quota
    def record(scores, ids, forced=None):
        value = original(scores, ids, forced); observed.append((len(ids), int(value.sum()))); return value
    monkeypatch.setattr(m, "exact_quota", record)
    monkeypatch.setattr(m, "BOOTSTRAP_REPLICATES", 30)
    result = m._paired_bootstrap(rows, np.arange(20), -np.arange(20), np.zeros(20, dtype=bool),
                                 np.zeros(20, dtype=bool), np.ones(20, dtype=bool))
    assert (2, 0) in observed and (38, 3) in observed and result["status"] == "PASS"
    assert len(observed) == 60
    assert all(observed[i] == observed[i+1] for i in range(0, len(observed), 2))


def test_bootstrap_never_silently_drops_infeasible_forced_failure_draws(monkeypatch):
    rows = [{"sample_id": f"i{i}", "premise_hash": "fail" if i < 2 else "ok"} for i in range(20)]
    failed = np.zeros(20, dtype=bool); failed[:2] = True
    monkeypatch.setattr(m, "BOOTSTRAP_REPLICATES", 30)
    result = m._paired_bootstrap(rows, np.arange(20), -np.arange(20), failed, np.zeros(20, dtype=bool), np.ones(20, dtype=bool))
    assert result["status"] == "UNAVAILABLE_FORCED_FAILURE_QUOTA" and result["ci95"] is None
    assert result["infeasible_draws"] > 0
