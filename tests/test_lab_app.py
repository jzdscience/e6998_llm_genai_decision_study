"""Presentation checks use explicit software fixtures, never scientific results."""
from copy import deepcopy
import math

import pytest

from decision_study.lab import app


def experiment():
    candidate = {"id": "gain", "label": "Gain", "fallback": "qwen", "family": "gain", "n": 4,
                 "correct": 3, "accuracy": .75, "deferrals": 0, "deferral_rate": 0., "rescue": 0,
                 "harm": 0, "target_budget": .1, "validation_deferral_rate": 0.,
                 "is_primary_candidate": True, "is_primary_reference": False}
    reference = {**candidate, "id": "confidence", "label": "Confidence", "family": "confidence",
                 "deferrals": 2, "deferral_rate": .5, "rescue": 1, "harm": 1,
                 "validation_deferral_rate": .1, "is_primary_candidate": False, "is_primary_reference": True}
    case = {"sample_id": "fixture-one", "state": '<script>alert("fixture")</script>', "gold": "yes", "labels": ["yes", "no"],
            "local": {"model": "startlux", "label": "StartLux", "prediction": "no", "confidence": .6,
                      "probabilities": {"yes": .4, "no": .6}},
            "fallbacks": {"qwen": {"model": "qwen", "label": "Qwen", "prediction": "yes"}},
            "selection_reason": "Software fixture only",
            "routes": {"gain": {"deferred": False, "final_prediction": "no", "correct": False, "outcome": "local_wrong"},
                       "confidence": {"deferred": True, "final_prediction": "yes", "correct": True, "outcome": "rescued"}}}
    return {"id": "v2", "title": "Software fixture", "task": "fixture", "n": 4, "formal_predictions": 8,
            "primary": {"label": "Gain minus confidence", "estimate_pp": 0., "lower_pp": -1., "upper_pp": 2.,
                        "candidate_id": "gain", "reference_id": "confidence", "confidence_level": .95,
                        "cluster_count": 2, "method": "software fixture"},
            "direct": [{"model": "startlux", "label": "StartLux", "n": 4, "correct": 3, "accuracy": .75}],
            "policies": [candidate, reference], "cases": [case], "case_note": "Software fixture only.",
            "notes": ["No model was called."], "provenance": {"software_fixture": True},
            "performance": {"measured": True, "note": "Software fixture only.", "rows": [
                {"condition": "local_only", "offered_rate": 10, "repetitions": 3, "goodput_mean": 9., "goodput_min": 8., "goodput_max": 10.},
                {"condition": "selected_cascade", "offered_rate": 10, "repetitions": 3, "goodput_mean": 4., "goodput_min": 3., "goodput_max": 5.},
                {"condition": "local_only", "offered_rate": 40, "repetitions": 3, "goodput_mean": 30., "goodput_min": 29., "goodput_max": 31.},
            ]}}


def test_case_switch_uses_selected_saved_route_without_mutating_evidence():
    exp = experiment()
    before = deepcopy(exp)
    local = app.replay_case(exp, "fixture-one", "gain")
    rescue = app.replay_case(exp, "fixture-one", "confidence")
    assert local["route"]["final_prediction"] == "no"
    assert local["route"]["deferred"] is False
    assert rescue["route"]["final_prediction"] == "yes"
    assert rescue["route"]["correct"] is True
    assert exp == before
    assert "did not use that output" in app.render_case(local)
    assert "used that output" in app.render_case(rescue)


def test_unknown_case_policy_or_missing_fallback_fails():
    exp = experiment()
    for sample, policy in [("unknown", "gain"), ("fixture-one", "unknown")]:
        with pytest.raises(ValueError):
            app.replay_case(exp, sample, policy)
    exp["cases"][0]["fallbacks"] = {}
    with pytest.raises(ValueError, match="fallback"):
        app.replay_case(exp, "fixture-one", "gain")


def test_same_family_names_distinguish_qwen_model_sizes_everywhere():
    exp = experiment()
    exp["id"] = "v3"
    for row, fallback in zip(exp["policies"], ("qwen4b", "qwen9b")):
        row["label"] = "Confidence"
        row["fallback"] = fallback
    exp["cases"][0]["fallbacks"] = {
        "qwen4b": {"model": "qwen4b", "label": "Qwen 4B", "prediction": "yes"},
        "qwen9b": {"model": "qwen9b", "label": "Qwen 9B", "prediction": "yes"},
    }
    assert [app.policy_label(row) for row in exp["policies"]] == ["Confidence · Qwen 4B", "Confidence · Qwen 9B"]
    html = app.render_experiment(exp)
    for label in ("Confidence · Qwen 4B", "Confidence · Qwen 9B"):
        assert html.count(label) == 2  # Policy table and validation/holdout budget table.
    assert "Confidence · Qwen 4B" in app.show_case(exp, "fixture-one", "gain")[0]
    assert "Confidence · Qwen 9B" in app.show_case(exp, "fixture-one", "confidence")[0]


def test_html_escapes_evidence_in_case_summary_and_ci():
    exp = experiment()
    attack = '<img src=x onerror="alert(1)">'
    exp["title"] = attack
    exp["direct"][0]["label"] = attack
    exp["policies"][0]["label"] = attack
    exp["notes"] = [attack]
    exp["primary"]["label"] = attack
    exp["id"] = attack
    exp["cases"][0]["gold"] = attack
    exp["cases"][0]["local"]["prediction"] = attack
    exp["cases"][0]["fallbacks"]["qwen"]["prediction"] = attack
    html = app.render_case(app.replay_case(exp, "fixture-one", "gain")) + app.render_experiment(exp) + app.render_primary_comparison([exp])
    assert "<script>" not in html
    assert "<img " not in html
    assert "&lt;script&gt;" in html
    assert "&lt;img" in html


def test_ci_axis_contains_zero_and_displays_percentage_points():
    exp = experiment()
    other = deepcopy(exp)
    other["id"] = "v3"
    html = app.render_primary_comparison([exp, other])
    assert 'x1="505" x2="505"' in html
    assert "[-1.000, +2.000] pp" in html
    assert "Both intervals include zero" in html
    assert "not pooled estimates" in html
    for value in (math.nan, math.inf, -math.inf):
        exp["primary"]["estimate_pp"] = value
        with pytest.raises(ValueError, match="non-finite"):
            app.render_primary_comparison([exp])


def test_actual_budgets_and_zero_handoff_are_not_replaced_with_target():
    html = app.render_experiment(experiment())
    assert "10.000%" in html
    assert "50.000%" in html
    assert "0.000%" in html
    assert "candidate chose no handoffs" in html
    assert "not a guaranteed holdout cap" in html


def test_performance_selects_only_saved_rate_and_uses_goodput():
    exp = experiment()
    html = app.render_performance(exp, 40)
    assert "30.000/s" in html
    assert "4.000/s" not in html
    assert "scheduled arrival" in html
    assert "not confidence intervals" in html
    with pytest.raises(ValueError, match="No saved performance"):
        app.render_performance(exp, 25)
    exp["performance"] = {"measured": False, "rows": [], "note": "Quality collection only."}
    html = app.render_performance(exp)
    assert "No co-resident performance measurement" in html
    assert "ds-perf-row" not in html


def test_nonlocal_launch_is_rejected_before_app_creation(monkeypatch):
    monkeypatch.setattr(app, "create_app", lambda *_: pytest.fail("Must not create an externally bound app"))
    with pytest.raises(ValueError, match="localhost"):
        app.launch("unused", "unused", host="0.0.0.0")
