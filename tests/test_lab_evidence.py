"""Protect presentation evidence from silent metric, route and identity changes."""
import copy
import hashlib
import json
import os
from pathlib import Path

import pytest

from decision_study.common import StudyError, canonical
from decision_study.lab import evidence


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    """Use the portable checkout artifact; optionally build from external audited data."""
    repository = Path(__file__).resolve().parents[1]
    portable = repository / "reports/decision-lab-2026-11-04"
    if (portable / "manifest.json").exists():
        return evidence.load_bundle(portable)
    root = os.environ.get("LAB_EVIDENCE_ROOT")
    if not root:
        pytest.skip("Install the portable Lab bundle or set LAB_EVIDENCE_ROOT to the audited external artifact tree")
    target = tmp_path_factory.mktemp("lab-evidence") / "portable"
    evidence.build(root, target)
    return evidence.load_bundle(target)


def _write(directory, data):
    directory.mkdir()
    content = canonical(data) + b"\n"
    (directory / "study.json").write_bytes(content)
    manifest = {"schema_version": 1, "kind": evidence.KIND,
                "files": {"study.json": {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}}}
    (directory / "manifest.json").write_bytes(canonical(manifest))
    return directory


def test_complete_result_counts_and_primary_intervals(recorded):
    v2, v3 = recorded["experiments"]["v2"], recorded["experiments"]["v3"]
    assert (v2["n"], v3["n"]) == (4000, 3200)
    assert (v2["formal_predictions"], v3["formal_predictions"]) == (10800, 17700)
    assert v2["primary"]["estimate_pp"] == pytest.approx(-0.225)
    assert v2["primary"]["lower_pp"] == pytest.approx(-0.9077920587048214)
    assert v2["primary"]["upper_pp"] == pytest.approx(0.4775369885651242)
    assert v3["primary"]["estimate_pp"] == pytest.approx(0.59375)
    assert v3["primary"]["lower_pp"] == pytest.approx(-0.5276719051845663)
    assert v3["primary"]["upper_pp"] == pytest.approx(1.6741242135245225)
    assert [v2["primary"]["cluster_count"], v3["primary"]["cluster_count"]] == [57, 2854]


def test_deferral_budget_is_actual_not_nominal(recorded):
    v2 = {p["id"]: p for p in recorded["experiments"]["v2"]["policies"]}
    v3 = {p["id"]: p for p in recorded["experiments"]["v3"]["policies"]}
    assert v2["confidence"]["validation_deferral_rate"] == 0.096
    assert v2["original_v1"]["validation_scope"] == "V1 original gate selection pool"
    assert v2["confidence"]["validation_scope"] == "V2 validation pool"
    assert v2["confidence"]["deferrals"] == 440
    assert v2["gain"]["deferrals"] == 383
    assert (v2["gain"]["rescue"], v2["gain"]["harm"]) == (0, 7)
    assert v3["qwen9b/confidence"]["validation_deferral_rate"] == 0.1
    assert v3["qwen9b/confidence"]["deferral_rate"] == 0.2321875
    assert (v3["qwen9b/confidence"]["rescue"], v3["qwen9b/confidence"]["harm"]) == (147, 166)
    for model in ("qwen4b", "qwen9b"):
        p = v3[model + "/positive_gain"]
        assert (p["decision_mode"], p["deferrals"], p["correct"]) == ("none", 0, 2374)


def test_measured_performance_has_no_v3_fabrication(recorded):
    v2 = recorded["experiments"]["v2"]["performance"]
    assert (v2["measured"], v2["duration_seconds"], v2["slo_ms"], len(v2["rows"])) == (True, 30, 500, 12)
    cascade = next(r for r in v2["rows"] if r["condition"] == "selected_cascade" and r["offered_rate"] == 40)
    assert cascade["goodput_mean"] == pytest.approx(4.888888888888889)
    assert cascade["repetitions"] == 3
    assert recorded["experiments"]["v3"]["performance"]["measured"] is False
    assert recorded["experiments"]["v3"]["performance"]["rows"] == []


def test_illustrative_cases_preserve_zero_route_and_reference_categories(recorded):
    for version, experiment in recorded["experiments"].items():
        assert len(experiment["cases"]) == 12
        assert {c["selection_reason"] for c in experiment["cases"]} == {
            "Local correct and retained", "Confidence reference rescues", "Confidence reference harms", "Both models wrong"}
        if version == "v3":
            for case in experiment["cases"]:
                for model in ("qwen4b", "qwen9b"):
                    decision = case["routes"][model + "/positive_gain"]
                    assert decision["deferred"] is False
                    assert decision["final_prediction"] == case["local"]["prediction"]


def test_payload_contains_no_machine_or_service_details(recorded):
    text = json.dumps(recorded)
    for forbidden in ("/Users/", "/home/", "zhanj289", "http://127.0.0.1", "GPU-", "sc1ns"):
        assert forbidden not in text


def test_loader_rejects_changed_payload_before_schema(tmp_path, monkeypatch):
    directory = _write(tmp_path / "portable", {"some": "data"})
    (directory / "study.json").write_text('{"some":"dAta"}\n')
    # Isolate this layer: no downstream schema check may rescue the hash guard.
    monkeypatch.setattr(evidence, "_validate_payload", lambda _data: None)
    with pytest.raises(StudyError, match="hash/size"):
        evidence.load_bundle(directory)


def test_rehashed_case_route_tamper_is_rejected(tmp_path, recorded):
    changed = copy.deepcopy(recorded)
    case = changed["experiments"]["v3"]["cases"][0]
    case["routes"]["qwen9b/positive_gain"]["deferred"] = True
    # The payload checksum is valid; the independent replay guard must reject it.
    directory = _write(tmp_path / "portable", changed)
    with pytest.raises(StudyError, match="Case route"):
        evidence.load_bundle(directory)


def test_missing_policy_is_rejected(recorded):
    changed = copy.deepcopy(recorded)
    changed["experiments"]["v2"]["policies"].pop()
    with pytest.raises(StudyError, match="Missing/duplicate policy"):
        evidence._validate_payload(changed)


def test_wrong_primary_delta_is_rejected(recorded):
    changed = copy.deepcopy(recorded)
    changed["experiments"]["v2"]["primary"]["estimate_pp"] = 1
    with pytest.raises(StudyError, match="Primary difference"):
        evidence._validate_payload(changed)


def test_nonfinite_primary_interval_is_rejected(recorded):
    changed = copy.deepcopy(recorded)
    changed["experiments"]["v3"]["primary"]["lower_pp"] = float("nan")
    with pytest.raises(StudyError, match="Invalid primary interval"):
        evidence._validate_payload(changed)


def test_unmeasured_performance_cannot_be_added(recorded):
    changed = copy.deepcopy(recorded)
    changed["experiments"]["v3"]["performance"]["rows"] = [{"goodput_mean": 12}]
    with pytest.raises(StudyError, match="no measured cascade"):
        evidence._validate_payload(changed)


def test_source_requires_completed_index_receipt_before_opening_data(tmp_path):
    (tmp_path / "records").mkdir()
    (tmp_path / "records/BUNDLE_INDEX.json").write_text('{"bundles":[]}')
    with pytest.raises(StudyError, match="Missing unique completed"):
        evidence._Source(tmp_path, "v3")


def test_no_overwrite_or_source_tree_output(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "keep.txt").write_text("user content")
    with pytest.raises(StudyError, match="already exists"):
        evidence.build(source, existing)
    assert (existing / "keep.txt").read_text() == "user content"
    with pytest.raises(StudyError, match="outside"):
        evidence.build(source, source / "new-output")


def test_source_paths_cannot_escape_or_follow_symlinks(tmp_path):
    with pytest.raises(StudyError, match="Unsafe"):
        evidence._safe(tmp_path, "../elsewhere")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (tmp_path / "link").symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(StudyError, match="Symlink"):
        evidence._safe(tmp_path, "link/data.json")
