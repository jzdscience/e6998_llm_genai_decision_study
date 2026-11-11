"""Guards for the portable follow-up evidence and the Follow-ups page of the Decision Lab.

The tracked portable bundle is the accepted evidence; tampered copies are written to temporary
directories only, so no frozen artifact is modified.
"""
import copy
import hashlib
import json
import math
import re
import shutil
from pathlib import Path

import pytest

from decision_study.common import StudyError, canonical
from decision_study.lab import followups

REPO = Path(__file__).resolve().parents[1]
PORTABLE = REPO / "reports/decision-lab-followups-2026-11-10"


@pytest.fixture(scope="module")
def recorded():
    return followups.load_bundle(PORTABLE)


def _write(directory: Path, data: dict) -> Path:
    directory.mkdir()
    # Same layout as canonical(), but NaN is allowed so the finite-number guard can be exercised.
    content = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=True).encode() + b"\n"
    (directory / "followups.json").write_bytes(content)
    manifest = {"schema_version": 1, "kind": followups.KIND,
                "files": {"followups.json": {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}}}
    (directory / "manifest.json").write_bytes(canonical(manifest))
    return directory


def _cell(data, condition, rate, repeat):
    return next(c for c in data["serving9b"]["cells"]
                if (c["condition"], c["rate"], c["repeat"]) == (condition, rate, repeat))


def _summary(data, condition, rate, slo):
    return next(r for r in data["serving9b"]["summary"]
                if (r["condition"], r["rate"], r["slo_ms"]) == (condition, rate, slo))


# Accepted evidence -------------------------------------------------------------------------------

def test_accepted_serving_inventory_totals_and_pass_sets(recorded):
    s = recorded["serving9b"]
    assert len(s["cells"]) == 36
    assert (s["totals"]["offered"], s["totals"]["successful"], s["totals"]["client_rejected"]) == (162801, 131935, 30866)
    assert s["totals"]["stage_predictions_checked"] == 140576
    assert s["totals"]["cascade_routes_replayed"] == 35860
    assert s["passing_rates"] == {"500": {"startlux4b": [10, 40], "startlux9b": [10, 20], "cascade": []},
                                  "1000": {"startlux4b": [10, 40], "startlux9b": [10, 20, 40], "cascade": [10]}}
    expected = {500: {"startlux4b": (7.258333, 14.85, 29.958333, 38.258333),
                      "startlux9b": (7.466667, 15.333333, 30.75, 0.291667),
                      "cascade": (6.95, 12.758333, 0.366667, 0.15)},
                1000: {"startlux4b": (7.258333, 14.85, 29.958333, 58.408333),
                       "startlux9b": (7.466667, 15.333333, 31.025, 0.841667),
                       "cascade": (7.308333, 14.558333, 0.908333, 0.25)}}
    for slo, conditions in expected.items():
        for condition, values in conditions.items():
            for rate, value in zip((10, 20, 40, 80), values):
                assert _summary(recorded, condition, rate, slo)["correct_goodput"]["median"] == pytest.approx(value, abs=5e-7)


def test_anomaly_repeat_is_retained_in_every_aggregate(recorded):
    s = recorded["serving9b"]
    anomaly = s["anomaly"]
    assert (anomaly["cell_id"], anomaly["late_500"], anomaly["client_rejections"]) == ("r0-startlux4b-rate20", 78, 1)
    assert anomaly["arrival_window_seconds"] == pytest.approx([110.6489, 114.2715], abs=1e-4)
    cell = _cell(recorded, "startlux4b", 20, 0)
    assert cell["slo"]["500"]["timely_fraction"] == 0.96875
    row = _summary(recorded, "startlux4b", 20, 500)
    assert row["timely_fraction"]["repeats"][0] == 0.96875
    # r0 stays in the aggregate: it is the first repeat value and lies inside the reported range.
    assert row["correct_goodput"]["repeats"][0] == cell["slo"]["500"]["correct_goodput"] == 15.05
    assert row["correct_goodput"]["min"] <= 15.05 <= row["correct_goodput"]["max"]
    assert row["passes_rule"] is False


def test_quality_and_memory_context_match_acceptance(recorded):
    q = {x["condition"]: x for x in recorded["serving9b"]["quality"]}
    assert q["startlux4b"]["pooled_live_accuracy"] == pytest.approx(0.741741, abs=1e-6)
    assert q["startlux4b"]["pooled_matched_historical_accuracy"] == pytest.approx(0.741237, abs=1e-6)
    assert q["cascade"]["pooled_matched_historical_accuracy"] == pytest.approx(0.743419, abs=1e-6)
    resources = {(r["condition"], r["role"]): r for r in recorded["serving9b"]["resources"]}
    assert resources[("startlux4b", "startlux4b")]["formal_kv_gib"] == [8.78]
    assert resources[("startlux9b", "startlux9b")]["formal_kv_gib"] == [4.22]
    assert resources[("cascade", "qwen9b")]["formal_kv_tokens"] == [112532]
    assert resources[("startlux9b", "startlux9b")]["preflight_kv_gib"] == 2.93
    pre = recorded["serving9b"]["preflight"]
    assert pre["pairwise_http_stage_overlap_seconds"] == pytest.approx(147.9986, abs=1e-4)
    assert pre["simultaneous_http_union_seconds"] == pytest.approx(14.7769, abs=1e-4)


def test_cases_disclose_selection_and_match_their_rules(recorded):
    cases = {c["kind"]: c for c in recorded["cases"]}
    assert set(cases) == set(followups.CASE_KINDS)
    assert "chosen by their known outcome" in recorded["case_note"]
    assert cases["upgrade_fixes"]["saved"]["deferred"] is False
    assert cases["fallback_fixes"]["saved"]["cascade_final"] == cases["fallback_fixes"]["gold"]
    assert cases["fallback_harms"]["saved"]["cascade_final"] != cases["fallback_harms"]["gold"]
    late = cases["late_correct"]
    assert late["trace"]["prediction"] == late["gold"] and late["trace"]["arrival_to_completion_ms"] > 1000
    assert all(c["pattern_base"] >= c["pattern_count"] >= 1 for c in cases.values())
    # Historical patterns count saved test items; the late case counts live requests, where inputs repeat.
    assert "out of those 3,200 inputs" in recorded["case_note"]
    assert "successful live requests in that cell" in recorded["case_note"]
    assert (late["pattern_count"], late["pattern_base"], late["pattern_unit"]) == (4282, 5746, "successful live requests in this cell")
    contexts = {k: (c["pattern_count"], c["context"]["count"]) for k, c in cases.items() if k != "late_correct"}
    assert contexts == {"upgrade_fixes": (95, 245), "fallback_fixes": (89, 164), "fallback_harms": (108, 157)}
    assert cases["fallback_harms"]["title"] == "The fallback breaks an answer both direct models got right"
    assert "arrival window" in recorded["serving9b"]["notes"][0]


def test_payload_contains_no_machine_or_service_details(recorded):
    text = json.dumps(recorded)
    for forbidden in ("/Users/", "/home/", "zhanj289", "http://127.0.0.1", "GPU-", "sc1ns", "slurm"):
        assert forbidden not in text


def test_old_v3_bundle_still_reports_no_cascade_performance():
    from decision_study.lab import evidence
    v3 = evidence.load_bundle(REPO / "reports/decision-lab-2026-11-04")["experiments"]["v3"]["performance"]
    assert v3["measured"] is False and v3["rows"] == []


# Tamper guards -----------------------------------------------------------------------------------

def test_changed_bytes_fail_the_checksum_before_any_semantic_check(tmp_path, monkeypatch, recorded):
    directory = _write(tmp_path / "portable", recorded)
    content = (directory / "followups.json").read_bytes().replace(b'"offered":162801', b'"offered":162802', 1)
    (directory / "followups.json").write_bytes(content)
    # Isolate the checksum layer: the semantic validator may not rescue it.
    monkeypatch.setattr(followups, "validate", lambda _data: None)
    with pytest.raises(StudyError, match="hash/size"):
        followups.load_bundle(directory)


@pytest.mark.parametrize("mutate, message", [
    (lambda d: _cell(d, "startlux9b", 40, 0)["slo"]["500"].__setitem__("correct_goodput", 31.5),
     "correct-and-timely responses per window second"),
    (lambda d: _cell(d, "cascade", 80, 1)["slo"]["1000"].__setitem__("timely_fraction", 0.5),
     "Timely fraction"),
    (lambda d: _cell(d, "startlux4b", 80, 2).__setitem__("rejected", 0), "count arithmetic"),
    (lambda d: _cell(d, "startlux9b", 10, 1).__setitem__("live_accuracy", float("nan")), "Non-finite"),
    (lambda d: d["serving9b"]["cells"].pop(), "36 measured cells"),
    (lambda d: _cell(d, "startlux4b", 40, 0).__setitem__("rate", 30), "36 measured cells"),
    (lambda d: d["serving9b"]["passing_rates"]["500"]["startlux4b"].append(20), "pass sets"),
    (lambda d: d["provenance"]["tracked_sources"].__setitem__(followups.SERVICE_AGGREGATES, "0" * 64), "Tracked source hashes"),
    (lambda d: d["provenance"]["bundles"]["serving9b"].__setitem__("sha256", "0" * 64), "Unknown source result archive"),
    (lambda d: next(c for c in d["cases"] if c["kind"] == "fallback_harms")["saved"].__setitem__("cascade_final", "neutral"),
     "Cascade label|selection rule"),
    (lambda d: d["cases"].clear(), "two to four cases"),
    (lambda d: next(c for c in d["cases"] if c["kind"] == "fallback_harms")["context"].__setitem__("count", 100),
     "context count"),
    (lambda d: next(c for c in d["cases"] if c["kind"] == "late_correct").__setitem__("pattern_unit", "saved test items"),
     "count unit"),
])
def test_rehashed_semantic_tampering_is_rejected(tmp_path, recorded, mutate, message):
    changed = copy.deepcopy(recorded)
    mutate(changed)
    directory = _write(tmp_path / "portable", changed)  # valid checksum; only the semantic guard can reject
    with pytest.raises(StudyError, match=message):
        followups.load_bundle(directory)


def test_consistent_change_to_correct_counts_is_rejected(tmp_path, recorded):
    """Even arithmetic that agrees with itself must match the accepted medians and summary."""
    changed = copy.deepcopy(recorded)
    for repeat in (0, 1, 2):
        cell = _cell(changed, "startlux9b", 80, repeat)
        m = cell["slo"]["500"]
        m["correct_met"] += 600
        m["met"] += 600
        m["correct_goodput"] = m["correct_met"] / 120
        m["timely_fraction"] = m["met"] / cell["offered"]
    followups._check_cell(cell)  # the change is internally consistent
    changed["serving9b"]["summary"] = followups._summary(changed["serving9b"]["cells"])
    with pytest.raises(StudyError, match="accepted"):
        followups.validate(changed)


def test_build_rejects_a_changed_tracked_summary(tmp_path):
    for relative in followups.EXPECTED_SOURCES:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / relative, target)
    followups._Tracked(tmp_path)  # the copies are the accepted bytes
    path = tmp_path / followups.SERVICE_AGGREGATES
    path.write_bytes(path.read_bytes().replace(b"93.4", b"94.4", 1))
    tracked = followups._Tracked(tmp_path)
    with pytest.raises(StudyError, match="Accepted summary changed"):
        followups._clinc_service(tracked)


def test_build_refuses_existing_or_in_tree_output(tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(StudyError, match="already exists"):
        followups.build(tmp_path, existing, REPO)
    with pytest.raises(StudyError, match="outside"):
        followups.build(tmp_path, tmp_path / "inside", REPO)


def test_missing_archive_is_reported_before_any_case_is_built(tmp_path):
    with pytest.raises(StudyError, match="archive not found"):
        followups._Archive(REPO, tmp_path)


# Display selectors -------------------------------------------------------------------------------

view = pytest.importorskip("decision_study.lab.followup_view")


def _best(html: str) -> str:
    return re.search(r"SLO, (.+?) returned the most correct answers", html).group(1)


@pytest.mark.parametrize("slo", [500, 1000])
@pytest.mark.parametrize("rate", [10, 20, 40, 80])
def test_every_measured_selection_renders_its_own_values(recorded, rate, slo):
    html = view.render_serving(recorded, rate, slo)
    for condition in ("startlux4b", "startlux9b", "cascade"):
        row = _summary(recorded, condition, rate, slo)
        assert f'{row["correct_goodput"]["median"]:.3f}' in html
        assert " / ".join(f"{v:.3f}" for v in row["correct_goodput"]["repeats"]) in html
    assert f"{slo:,} ms" in html and f"offered {rate}/s" in html
    overview = view.render_serving_overview(recorded, slo, rate)
    assert f"{rate}/s (selected)" in overview


@pytest.mark.parametrize("rate, slo", [(30, 500), (0, 500), (40.5, 500), ("80x", 500), (True, 500), (None, 500),
                                       (40, 750), (40, 0)])
def test_unmeasured_selection_is_rejected_without_interpolation(recorded, rate, slo):
    with pytest.raises(ValueError, match="No measured"):
        view.render_serving(recorded, rate, slo)


def test_rate_change_from_40_to_80_reverses_the_ranking(recorded):
    at40 = view.render_serving(recorded, 40, 500)
    at80 = view.render_serving(recorded, 80, 500)
    assert _best(at40) == "StartLux-9B direct"
    assert "missed the 99% on-time rule in at least one repeat (worst 98.87%)" in at40
    assert _best(at80) == "StartLux-4B direct"
    assert "38.258" in at80 and "No deployment met the 99% on-time rule" in at80


def test_slo_change_updates_values_and_pass_status(recorded):
    strict, relaxed = view.render_serving(recorded, 40, 500), view.render_serving(recorded, 40, 1000)
    assert "30.750" in strict and "31.025" in relaxed
    assert "missed the 99%" in strict and "missed the 99%" not in relaxed
    assert "StartLux-9B: 10, 20;" in view.render_serving_overview(recorded, 500)
    assert "StartLux-9B: 10, 20, 40;" in view.render_serving_overview(recorded, 1000)
    assert "cascade: none" in view.render_serving_overview(recorded, 500)


def test_anomaly_note_appears_at_20_only(recorded):
    assert "kept as measured" in view.render_serving(recorded, 20, 500)
    assert all("kept as measured" not in view.render_serving(recorded, r, 500) for r in (10, 40, 80))


def test_default_choices_exist_in_the_evidence(recorded):
    assert 40 in recorded["serving9b"]["rates"] and 500 in recorded["serving9b"]["slos_ms"]
    assert recorded["clinc_service"]["rates"][-1] == 100
    assert recorded["cases"][0]["id"] in {c["id"] for c in recorded["cases"]}


@pytest.mark.parametrize("rate", [5, 25, 100])
def test_clinc_service_selector_uses_measured_rates_only(recorded, rate):
    html = view.render_clinc_service(recorded, rate)
    assert f"Offered load {rate} requests/s" in html
    for row in (r for r in recorded["clinc_service"]["rows"] if r["rate"] == rate):
        assert f'{row["correct_goodput"]["median"]:.2f}' in html
    with pytest.raises(ValueError, match="No measured"):
        view.render_clinc_service(recorded, 50)


def test_cases_render_and_unknown_case_fails(recorded):
    for case in recorded["cases"]:
        html, ids = view.show_followup_case(recorded, case["id"])
        assert case["sample_id"] == ids["sample_id"] and "nothing here sends a new request" in html
    late = view.show_followup_case(recorded, "late-correct")[0]
    assert "adds nothing to correct goodput" in late
    assert "4,282 of 5,746 successful live requests in this cell" in late
    harms = view.show_followup_case(recorded, "fallback-harms")[0]
    assert "108 of 3,200 saved test items share this pattern, within 157 saved test items the frozen cascade harms" in harms
    with pytest.raises(ValueError, match="saved follow-up cases"):
        view.show_followup_case(recorded, "unknown")


def test_html_escapes_evidence_text(recorded):
    attack = '<img src=x onerror="alert(1)">'
    changed = copy.deepcopy(recorded)
    for d in changed["serving9b"]["deployments"]:
        d["label"] = d["short"] = d["gpu_share"] = attack
    changed["serving9b"]["notes"] = [attack]
    for case in changed["cases"]:
        case["state"] = case["title"] = attack
    html = (view.render_serving(changed, 40, 500) + view.render_serving_overview(changed, 500, 40)
            + view.render_serving_context(changed) + "".join(view.show_followup_case(changed, c["id"])[0] for c in changed["cases"]))
    assert "<img " not in html and "&lt;img" in html


def test_followups_tab_is_added_without_touching_the_original_tabs():
    gr = pytest.importorskip("gradio")
    from decision_study.lab import app
    built = app.create_app(REPO / "reports/decision-lab-2026-11-04", REPO / "reports/decision-lab-2026-11-02", PORTABLE)
    tabs = [b.label for b in built.blocks.values() if isinstance(b, gr.Tab)]
    assert {"Overview", "V2 · SQuAD", "V3 · ANLI", "Follow-ups", "V1 replay"} <= set(tabs)
    plain = app.create_app(REPO / "reports/decision-lab-2026-11-04", REPO / "reports/decision-lab-2026-11-02")
    assert "Follow-ups" not in [b.label for b in plain.blocks.values() if isinstance(b, gr.Tab)]
