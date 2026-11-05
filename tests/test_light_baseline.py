"""Artificial software fixtures only; these are not model-study observations."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("sklearn", reason="Install the optional baseline CPU environment")

from decision_study.baselines import study


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root / "configs/clinc_tfidf400_protocol.json").read_text())
    labels = protocol["labels"]
    v1, v2, output = (tmp_path / name for name in ("legacy-v1", "legacy-v2", "baseline"))
    rows, gold = [], {}
    marker = {"calibration": "trainmarker", "validation": "validationonlymarker",
              "test": "testonlymarker", "stress": "stressonlymarker"}
    for split, per_intent, oos_count in [("calibration", 20, 100), ("validation", 20, 100),
                                        ("test", 30, 150), ("stress", 0, 850)]:
        for class_index, label in enumerate(labels):
            count = oos_count if label == "oos" else per_intent
            for index in range(count):
                sample_id = f"fictional-clinc:{split}:{class_index}:{index}"
                text = f"ARTIFICIAL fixture intentword{class_index} {marker[split]} entry{index}"
                rows.append({"sample_id": sample_id, "task": "clinc", "split": split,
                             "state": text, "group_id": sample_id, "labels": list(labels),
                             "criteria": {name: f"fictional {name}" for name in labels},
                             "instructions": "Artificial software-test input only.",
                             "source_split": split})
                gold[sample_id] = label
    fitting = [row for row in rows if row["split"] in ("calibration", "validation")]
    write_rows(v1 / "data/examples.jsonl", rows)
    write_json(v1 / "data/gold.json", gold)
    write_rows(v2 / "data/examples.jsonl", fitting)
    write_json(v2 / "data/fit_gold.json", {row["sample_id"]: gold[row["sample_id"]] for row in fitting})
    manifest = {"protocol": "solo-v1", "fingerprint": "fixture-only-not-real-data",
                "label_order": {"clinc": labels}, "counts": {"clinc": protocol["counts"]},
                "files": {"examples.jsonl": sha(v1 / "data/examples.jsonl"),
                          "gold.json": sha(v1 / "data/gold.json")}}
    write_json(v1 / "data/manifest.json", manifest)
    protocol_path = tmp_path / "fixture-protocol.json"
    result = SimpleNamespace(v1=v1, v2=v2, output=output, protocol=protocol,
                             protocol_path=protocol_path, rows=rows, gold=gold, labels=labels)

    def repin():
        manifest = json.loads((v1 / "data/manifest.json").read_text())
        manifest["files"] = {"examples.jsonl": sha(v1 / "data/examples.jsonl"),
                             "gold.json": sha(v1 / "data/gold.json")}
        write_json(v1 / "data/manifest.json", manifest)
        paths = {"v1_manifest_sha256": v1 / "data/manifest.json",
                 "v1_examples_sha256": v1 / "data/examples.jsonl",
                 "v1_gold_sha256": v1 / "data/gold.json",
                 "v2_examples_sha256": v2 / "data/examples.jsonl",
                 "v2_fit_gold_sha256": v2 / "data/fit_gold.json"}
        pins = {key: sha(path) for key, path in paths.items()}
        protocol["sources"] = pins
        monkeypatch.setattr(study, "SOURCE_PINS", pins)
        write_json(protocol_path, protocol)
    result.repin = repin
    repin()
    return result


def fit(fixture):
    return study.fit(fixture.v1, fixture.v2, fixture.output, protocol_path=fixture.protocol_path)


def evaluate(fixture):
    return study.evaluate(fixture.v1, fixture.v2, fixture.output, protocol_path=fixture.protocol_path)


def forbid_gold(monkeypatch, fixture):
    original = Path.open
    forbidden = (fixture.v1 / "data/gold.json").resolve()

    def guarded(path, *args, **kwargs):
        if path.resolve() == forbidden:
            raise AssertionError("Forbidden V1 evaluation-gold read, including hashing")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)


def snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_fit_never_reads_or_hashes_v1_gold_and_preserves_sources(fixture, monkeypatch):
    originals = snapshot(fixture.v1), snapshot(fixture.v2)
    with monkeypatch.context() as guarded:
        forbid_gold(guarded, fixture)
        fit(fixture)
    assert (snapshot(fixture.v1), snapshot(fixture.v2)) == originals
    assert (fixture.output / "freeze.json").is_file()
    assert not (fixture.output / "evaluation-access.json").exists()
    assert not (fixture.output / "evaluation.json").exists()


def test_vocabulary_and_idf_fit_only_calibration(fixture, monkeypatch):
    from sklearn.feature_extraction.text import TfidfVectorizer
    observed = []
    original = TfidfVectorizer.fit_transform

    def record(self, documents, *args, **kwargs):
        documents = list(documents)
        observed.append(documents)
        matrix = original(self, documents, *args, **kwargs)
        assert "trainmarker" in self.vocabulary_
        assert all(token not in self.vocabulary_ for token in
                   ["validationonlymarker", "testonlymarker", "stressonlymarker"])
        return matrix
    monkeypatch.setattr(TfidfVectorizer, "fit_transform", record)
    fit(fixture)
    assert len(observed) == 1
    assert len(observed[0]) == 400
    assert all("trainmarker" in text for text in observed[0])


@pytest.mark.parametrize("artifact", ["freeze.json", "model.json", "predictions.jsonl"])
def test_tampered_or_missing_freeze_evidence_rejected_before_gold(fixture, monkeypatch, artifact):
    fit(fixture)
    target = fixture.output / artifact
    if artifact == "freeze.json":
        target.unlink()
    else:
        target.write_bytes(target.read_bytes() + b" \n")
    with monkeypatch.context() as guarded:
        forbid_gold(guarded, fixture)
        with pytest.raises((ValueError, RuntimeError)):
            evaluate(fixture)
    assert not (fixture.output / "evaluation-access.json").exists()


def test_evaluation_never_refits_and_records_access_before_gold(fixture, monkeypatch):
    fit(fixture)
    original_open = Path.open
    gold_path = (fixture.v1 / "data/gold.json").resolve()
    reads = []

    def observed(path, *args, **kwargs):
        if path.resolve() == gold_path:
            assert (fixture.output / "evaluation-access.json").is_file()
            assert (fixture.output / "freeze.json").is_file()
            reads.append(str(path))
        return original_open(path, *args, **kwargs)

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("Evaluation tried to refit")
    monkeypatch.setattr(Path, "open", observed)
    monkeypatch.setattr(study, "_fit_model", forbidden_fit)
    evaluate(fixture)
    assert reads
    assert (fixture.output / "evaluation.json").is_file()


def test_source_hash_mismatch_rejected_before_fitting(fixture, monkeypatch):
    path = fixture.v2 / "data/fit_gold.json"
    path.write_bytes(path.read_bytes() + b"\n")
    forbid_gold(monkeypatch, fixture)
    with pytest.raises((ValueError, RuntimeError)):
        fit(fixture)
    assert not (fixture.output / "freeze.json").exists()


@pytest.mark.parametrize("problem", ["missing-row", "duplicate-id"])
def test_source_population_guards_survive_recomputed_file_hashes(fixture, monkeypatch, problem):
    rows = deepcopy(fixture.rows)
    if problem == "missing-row":
        rows.pop()
    else:
        rows[1]["sample_id"] = rows[0]["sample_id"]
    write_rows(fixture.v1 / "data/examples.jsonl", rows)
    fixture.repin()
    forbid_gold(monkeypatch, fixture)
    with pytest.raises((ValueError, RuntimeError)):
        fit(fixture)
    assert not (fixture.output / "freeze.json").exists()


def test_fit_reuse_is_byte_identical_and_partial_run_is_not_overwritten(fixture):
    fit(fixture)
    before = snapshot(fixture.output)
    fit(fixture)
    assert snapshot(fixture.output) == before
    (fixture.output / "freeze.json").unlink()
    incomplete = snapshot(fixture.output)
    with pytest.raises((ValueError, RuntimeError)):
        fit(fixture)
    assert snapshot(fixture.output) == incomplete


@pytest.mark.parametrize("target", ["v1", "v2"])
def test_output_cannot_overlap_input_roots(fixture, target):
    root = getattr(fixture, target)
    before = snapshot(root)
    with pytest.raises((ValueError, RuntimeError)):
        study.fit(fixture.v1, fixture.v2, root / "new-output", protocol_path=fixture.protocol_path)
    assert snapshot(root) == before


def test_symlink_output_rejected_without_writes(fixture, tmp_path):
    external = tmp_path / "external"
    external.mkdir()
    (external / "sentinel").write_text("unchanged")
    fixture.output.symlink_to(external, target_is_directory=True)
    before = snapshot(external)
    with pytest.raises((ValueError, RuntimeError)):
        fit(fixture)
    assert snapshot(external) == before


def test_active_output_lock_blocks_fitting(fixture):
    import fcntl
    fixture.output.mkdir()
    with (fixture.output / ".baseline.lock").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises((ValueError, RuntimeError)):
            fit(fixture)
    assert not (fixture.output / "freeze.json").exists()


def test_convergence_warning_is_a_failure(fixture, monkeypatch):
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    original = LogisticRegression.fit

    def nonconvergent(self, *args, **kwargs):
        fitted = original(self, *args, **kwargs)
        warnings.warn("Artificial convergence failure", ConvergenceWarning)
        return fitted
    monkeypatch.setattr(LogisticRegression, "fit", nonconvergent)
    with pytest.raises((ConvergenceWarning, ValueError, RuntimeError)):
        fit(fixture)
    assert not (fixture.output / "freeze.json").exists()


def test_label_remapping_and_stress_workload_denominators(fixture, monkeypatch):
    """An artificial all-OOS predictor exposes ordering and denominator mistakes."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    def all_oos(self, matrix):
        probabilities = np.zeros((matrix.shape[0], len(self.classes_)))
        probabilities[:, list(self.classes_).index("oos")] = 1.0
        return probabilities
    monkeypatch.setattr(LogisticRegression, "predict_proba", all_oos)
    fit(fixture)
    rows = [json.loads(line) for line in (fixture.output / "predictions.jsonl").read_text().splitlines()]
    assert len(rows) == 1850
    assert all(row["predicted_label"] == "oos" for row in rows)
    assert all(row["probabilities"][-1] == 1.0 and sum(row["probabilities"]) == 1.0 for row in rows)
    result = evaluate(fixture)
    metrics = result["metrics"]
    assert metrics["validation"]["count"] == 400
    assert metrics["main_test"]["count"] == 600
    assert metrics["oos_only_stress"]["count"] == 850
    assert metrics["historical_mixed_stress"]["count"] == 1300
    assert metrics["main_test"]["accuracy"] == pytest.approx(150 / 600)
    assert metrics["oos_only_stress"]["oos_recall"] == 1.0
    assert "accuracy" not in metrics["oos_only_stress"]
    assert "macro_f1" not in metrics["oos_only_stress"]
    assert metrics["historical_mixed_stress"]["accuracy"] == pytest.approx(850 / 1300)
    assert metrics["main_test"]["banking_accuracy"] == 0.0


@pytest.mark.parametrize("problem", ["nan", "bad-sum", "negative", "wrong-shape"])
def test_invalid_model_probabilities_never_get_frozen(fixture, monkeypatch, problem):
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    def malformed(self, matrix):
        columns = len(self.classes_) - (problem == "wrong-shape")
        probabilities = np.full((matrix.shape[0], columns), 1 / columns)
        if problem == "nan":
            probabilities[0, 0] = np.nan
        elif problem == "bad-sum":
            probabilities[0] *= 0.5
        elif problem == "negative":
            probabilities[0, 0] = -0.1
            probabilities[0, 1] += 0.1 + 1 / columns
        return probabilities
    monkeypatch.setattr(LogisticRegression, "predict_proba", malformed)
    with pytest.raises((ValueError, RuntimeError)):
        fit(fixture)
    assert not (fixture.output / "freeze.json").exists()


def test_partial_evaluation_is_preserved_without_second_gold_access(fixture, monkeypatch):
    fit(fixture)
    evaluate(fixture)
    (fixture.output / "evaluation.json").unlink()
    before = snapshot(fixture.output)
    forbid_gold(monkeypatch, fixture)
    with pytest.raises((ValueError, RuntimeError), match="access|partial"):
        evaluate(fixture)
    assert snapshot(fixture.output) == before


def test_bundle_refuses_tampered_model_even_after_evaluation(fixture):
    fit(fixture)
    evaluate(fixture)
    path = fixture.output / "model.json"
    path.write_bytes(path.read_bytes() + b" \n")
    with pytest.raises((ValueError, RuntimeError)):
        study.bundle(fixture.output)
    assert not (fixture.output / "exports/light_baseline_bundle.zip").exists()


def test_probability_validator_rejects_nonfinite_vectors_on_its_own(fixture):
    context = study._context(fixture.v1, fixture.v2, fixture.protocol_path)
    probabilities = [[1 / 16] * 16 for _ in context["predict"]]
    probabilities[-1][0] = float("nan")
    with pytest.raises((ValueError, RuntimeError), match="probability"):
        study._predictions(context, probabilities)


@pytest.mark.parametrize("identity", [{"git_available": False, "git_dirty": None},
                                      {"git_available": True, "git_dirty": True}])
def test_cli_requires_a_versioned_clean_checkout(monkeypatch, identity):
    monkeypatch.setattr(study, "_source_identity", lambda: identity)
    with pytest.raises(study.BaselineError):
        study._require_clean_checkout()


def test_cli_run_builds_a_reusable_complete_artifact_bundle(fixture, monkeypatch, capsys):
    import zipfile
    # Guard behavior is independently tested above; this fixture exercises CLI
    # parsing plus the complete real fit/evaluate/export software pipeline.
    monkeypatch.setattr(study, "_require_clean_checkout", lambda: None)
    monkeypatch.setattr(study, "PROTOCOL_PATH", fixture.protocol_path)
    args = ["run", "--v1-dir", str(fixture.v1), "--v2-dir", str(fixture.v2),
            "--output-dir", str(fixture.output)]
    assert study.main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["evaluation"]["status"] == "COMPLETE"
    model = json.loads((fixture.output / "model.json").read_text())
    fit_receipt = json.loads((fixture.output / "fit.json").read_text())
    assert fit_receipt["feature_count"] == len(model["vocabulary"])
    assert fit_receipt["model_bytes"] == (fixture.output / "model.json").stat().st_size
    bundle = Path(result["bundle"])
    original_bundle = bundle.read_bytes()
    with zipfile.ZipFile(bundle) as archive:
        assert set(archive.namelist()) == {"protocol.json", "model.json", "predictions.jsonl",
                                           "fit.json", "freeze.json", "evaluation-access.json",
                                           "evaluation.json"}
        for name in archive.namelist():
            assert archive.read(name) == (fixture.output / name).read_bytes()
    assert study.main(args) == 0
    capsys.readouterr()
    assert bundle.read_bytes() == original_bundle


def test_export_respects_an_existing_run_lock(fixture):
    fit(fixture)
    evaluate(fixture)
    lock = fixture.output / ".baseline.lock"
    lock.write_text("Artificial active-owner fixture\n")
    with pytest.raises(study.BaselineError, match="locked"):
        study.bundle(fixture.output)
    assert lock.read_text() == "Artificial active-owner fixture\n"
    assert not (fixture.output / "exports/light_baseline_bundle.zip").exists()
