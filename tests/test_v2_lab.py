"""Behavior and evidence tests for the CPU Decision Lab, without GPU or network."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from decision_study.v2.lab import bundle
from decision_study.v2.lab.app import render_case_panels, render_sandbox_summary

ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT / "results/returned-experiment-2026-11-02-101054"
BENCHMARK = ROOT / "results/returned-benchmark-2026-11-02-130735"


def toy_data():
    """An explicitly test-only in-memory fixture; never exported or displayed."""
    task = {"labels": ["yes", "no"], "criteria": {"yes": "yes", "no": "no"}, "instructions": "Test fixture"}
    gate = {"threshold": .5, "accept_none": False}
    cal = {"tasks": {t: {"temperature": {"temperature": 1.0}, "gates": {c: deepcopy(gate) for c in ("released", "refitted")}} for t in ("clinc", "squad")}}
    samples = {}
    for sid, p, local_label, qwen_label, gold in (("a", .5, "yes", "no", "yes"), ("b", .9, "yes", "no", "no")):
        local = {"status": "ok", "prediction": local_label, "probabilities": {"yes": p, "no": 1-p}}
        samples[sid] = {"sample_id": sid, "task": "squad", "split": "test", "group_id": sid,
                        "state": "TEST FIXTURE", "gold": gold, "predictions": {"kev": deepcopy(local), "startlux": deepcopy(local),
                        "qwen": {"status": "ok", "prediction": qwen_label, "probabilities": None}}}
    return {"tasks": {"clinc": deepcopy(task), "squad": deepcopy(task)}, "samples": samples,
            "calibrations": {"kev": deepcopy(cal), "startlux": deepcopy(cal)}}


@pytest.fixture(scope="module")
def real_lab(tmp_path_factory):
    if not (FORMAL / "bundle_manifest.json").exists() or not (BENCHMARK / "bundle_manifest.json").exists():
        pytest.skip("Archived v1 evidence is not installed; pure handler tests still run")
    parent = tmp_path_factory.mktemp("portable-lab")
    path = parent / "bundle"
    bundle.build(FORMAL, BENCHMARK, path)
    return path, bundle.load_bundle(path)


def test_exact_tie_accepted_and_upper_threshold_defers():
    data = toy_data()
    accepted = bundle.sample_replay(data, "startlux", "a")
    assert accepted["confidence"] == .5 and accepted["accepted"] is True
    assert accepted["final_prediction"] == "yes"
    changed = bundle.sandbox(data, "startlux", "squad", threshold=.500001)
    assert changed["accepted"] == 1
    assert changed["harmed"] == 1


def test_accept_none_is_not_threshold_one():
    data = toy_data()
    data["samples"]["a"]["predictions"]["startlux"]["probabilities"] = {"yes": 1.0, "no": 0.0}
    data["calibrations"]["startlux"]["tasks"]["squad"]["temperature"]["temperature"] = .05
    at_one = bundle.sandbox(data, "startlux", "squad", threshold=1.0)
    accept_none = bundle.sandbox(data, "startlux", "squad", threshold=1.0, accept_none=True)
    assert at_one["accepted"] > 0
    assert accept_none["accepted"] == 0
    assert accept_none["selective_risk"] is None
    assert accept_none["cascade_accuracy"] == accept_none["qwen_accuracy"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1, True])
def test_invalid_threshold_rejected(value):
    with pytest.raises(ValueError, match="Threshold"):
        bundle.sandbox(toy_data(), "startlux", "squad", threshold=value)


def test_unknown_input_and_missing_split_are_not_imputed():
    with pytest.raises(ValueError, match="Unknown sample"):
        bundle.sample_replay(toy_data(), "startlux", "not-recorded")
    with pytest.raises(ValueError, match="main/stress"):
        bundle.sandbox(toy_data(), "startlux", "squad", "validation")
    with pytest.raises(ValueError, match="Choose"):
        bundle.sample_replay(toy_data(), "qwen", "a")


def test_saved_evidence_is_not_mutated_by_exploration():
    data = toy_data()
    before = deepcopy(data)
    result = bundle.sandbox(data, "kev", "squad", threshold=.92)
    replay = bundle.sample_replay(data, "kev", "a", "released")
    assert result["performance_estimate"] is None
    assert not result["matches_frozen_gate"]
    assert replay["qwen_probabilities"] is None
    assert replay["qwen_used_by_policy"] is False
    assert data == before


def test_public_evidence_text_is_escaped_in_all_html_panels():
    data = toy_data()
    attack = '<img src=x onerror="alert(1)"><script>alert(2)</script>'
    data["samples"]["a"]["state"] = f"Passage:\n{attack}\n\nQuestion:\n{attack}"
    result = bundle.sample_replay(data, "startlux", "a")
    result["local_prediction"] = result["qwen_prediction"] = result["final_prediction"] = result["gold"] = attack
    result["probabilities"] = {attack: .5, "safe": .5}
    html = "\n".join(render_case_panels(result))
    assert "<script>" not in html and "<img" not in html
    assert "&lt;script&gt;" in html and "&lt;img" in html


def test_empty_acceptance_display_is_explicit():
    result = bundle.sandbox(toy_data(), "startlux", "squad", accept_none=True)
    assert "Not defined" in render_sandbox_summary(result)


def test_full_archive_counts_and_frozen_golden_results(real_lab):
    _, data = real_lab
    assert sum(data["metrics"]["prediction_counts"].values()) == 15950
    assert len(data["samples"]) == 3450
    assert len(data["benchmark_cells"]) == 60
    squad = bundle.sandbox(data, "startlux", "squad")
    assert (squad["n"], squad["accepted"], squad["accepted_errors"], squad["deferral_count"]) == (2000, 1875, 118, 125)
    assert (squad["rescued"], squad["harmed"], squad["net_correct"]) == (29, 24, 5)
    assert squad["local_accuracy"] == .9125 and squad["cascade_accuracy"] == .915
    clinc = bundle.sandbox(data, "startlux", "clinc")
    assert (clinc["n"], clinc["accepted"], clinc["accepted_errors"]) == (600, 599, 38)
    assert clinc["cascade_accuracy"] == .935
    kev = bundle.sandbox(data, "kev", "squad")
    assert (kev["accepted"], kev["accepted_errors"], kev["rescued"], kev["harmed"]) == (708, 31, 184, 94)
    assert kev["cascade_accuracy"] == .8585


def test_stress_workload_reuses_banking_correctly(real_lab):
    _, data = real_lab
    main = bundle.workload_samples(data, "clinc")
    stress = bundle.workload_samples(data, "clinc", "stress")
    assert len(stress) == 1300
    common = {r["sample_id"] for r in main} & {r["sample_id"] for r in stress}
    assert len(common) == 450
    assert all(r["gold"] != "oos" for r in stress if r["sample_id"] in common)


def test_four_curated_replays_match_existing_evidence(real_lab):
    _, data = real_lab
    expected = {
        "local_helps": (True, True, "account_blocked"),
        "qwen_rescue": (False, True, "unanswerable"),
        "fallback_harms": (False, False, "answerable"),
        "confident_miss": (True, False, "unanswerable"),
    }
    assert len(data["cases"]) == 4
    for case in data["cases"]:
        r = bundle.sample_replay(data, "startlux", case["sample_id"])
        assert (r["accepted"], r["correct"], r["final_prediction"]) == expected[case["category"]]


def test_measured_performance_does_not_use_formal_deferral(real_lab):
    _, data = real_lab
    rows = {row["deployment"]: row for row in bundle.measured_performance(data, "squad", 8)}
    cascade = rows["startlux_to_qwen"]
    assert cascade["rounds"] == 3 and cascade["unique_inputs"] == 200
    assert cascade["fallback_fraction"] == pytest.approx(.05)
    assert cascade["throughput"] == pytest.approx(44.20801348856926)
    assert cascade["p95_ms"] == pytest.approx(249.6728345751674)


def test_bundle_has_no_private_runtime_paths(real_lab):
    path, data = real_lab
    serialized = (path / "lab.json").read_text()
    for value in ("/Users/", "/home/", "zhanj289", "http://127.0.0.1", "source_row_session"):
        assert value not in serialized
    assert all(s["predictions"]["qwen"]["probabilities"] is None for s in data["samples"].values())


def test_modified_portable_bundle_fails_checksum(real_lab, tmp_path):
    path, _ = real_lab
    (tmp_path / "manifest.json").write_bytes((path / "manifest.json").read_bytes())
    (tmp_path / "lab.json").write_bytes((path / "lab.json").read_bytes() + b" ")
    with pytest.raises(ValueError, match="checksum"):
        bundle.load_bundle(tmp_path)


def test_rehashed_but_invalid_schema_is_rejected(real_lab, tmp_path):
    _, data = real_lab
    changed = deepcopy(data)
    changed["schema_version"] = 999
    body = json.dumps(changed).encode()
    (tmp_path / "lab.json").write_bytes(body)
    (tmp_path / "manifest.json").write_text(json.dumps({"schema_version": 1, "kind": "decision-lab", "files": {"lab.json": hashlib.sha256(body).hexdigest()}}))
    with pytest.raises(ValueError, match="schema"):
        bundle.load_bundle(tmp_path)


def test_output_cannot_be_written_into_frozen_input_tree():
    with pytest.raises(ValueError, match="outside"):
        bundle.build(FORMAL, BENCHMARK, FORMAL / "this-output-must-never-exist")


def test_create_app_is_lazy_and_has_real_handlers(real_lab):
    pytest.importorskip("gradio")
    from decision_study.v2.lab.app import create_app
    path, _ = real_lab
    app = create_app(path)
    assert not app.is_running
    names = {fn.api_name for fn in app.fns.values()}
    assert {"replay_case", "browse_case", "explore_threshold", "measured_performance"}.issubset(names)
    app.close()
    app._decision_lab_exports.cleanup()
