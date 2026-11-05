"""Artificial software fixtures, never experimental quality observations."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn", reason="Use the isolated cheap-baseline environment")

from decision_study.cheap_baselines import study, encoder, data

REAL_CONTEXT = study._context


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(path):
    return {str(p.relative_to(path)): p.read_bytes() for p in path.rglob("*") if p.is_file()}


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    protocol = json.loads((Path(__file__).resolve().parents[1] /
                           "configs/clinc_cheap_protocol.json").read_text())
    labels = protocol["labels"]
    v1, v2, prepared, model, output = (tmp_path / x for x in
                                      ("v1", "v2", "prepared", "encoder", "output"))
    for root in (v1, v2, prepared, model):
        root.mkdir()
        (root / "unchanged.txt").write_text("Artificial fixture sentinel\n")
    roles = {role: [] for role in ("calibration", "validation", "test", "stress")}
    full, gold = [], {}
    for class_index, label in enumerate(labels):
        for index in range(100):
            text = f"ARTIFICIAL classword{class_index} trainonlymarker entry{index}"
            source = "oos_train" if label == "oos" else "train"
            source_index = index if label == "oos" else class_index * 100 + index
            sid = f"clinc:{source}:{source_index}:{hashlib.sha256(text.encode()).hexdigest()[:16]}"
            row = {"sample_id": sid, "state": text, "label": label, "source_split": source}
            full.append(row)
            gold[sid] = label
            if label == "oos" or index < 20:
                roles["calibration"].append({k: v for k, v in row.items() if k != "label"} |
                                             {"split": "calibration", "task": "clinc", "labels": labels})
        for role, per_bank, oos in (("validation", 20, 100), ("test", 30, 150), ("stress", 0, 850)):
            for index in range(oos if label == "oos" else per_bank):
                sid = f"artificial:{role}:{class_index}:{index}"
                roles[role].append({"sample_id": sid, "task": "clinc", "split": role,
                                    "state": f"ARTIFICIAL classword{class_index} {role}onlymarker entry{index}",
                                    "labels": labels, "source_split": role})
                gold[sid] = label
    full.sort(key=lambda row: row["sample_id"])
    for rows in roles.values():
        rows.sort(key=lambda row: row["sample_id"])
    lookup = {row["sample_id"]: row for row in full}
    train = {"400": [lookup[row["sample_id"]] for row in roles["calibration"]], "1600": full}
    predict = roles["validation"] + roles["test"] + roles["stress"]
    dump(v1 / "data/gold.json", gold)
    gold_sha = sha(v1 / "data/gold.json")
    fit_gold = {row["sample_id"]: gold[row["sample_id"]]
                for role in ("calibration", "validation") for row in roles[role]}
    sources = {"v1_gold_sha256": gold_sha}
    protocol_path = tmp_path / "fixture-protocol.json"
    dump(protocol_path, protocol)
    context = {"protocol": protocol, "protocol_sha256": sha(protocol_path),
               "environment": {"synthetic_fixture": True},
               "source_identity": {"synthetic_fixture": True},
               "model_identity": {"synthetic_fixture": True},
               "model_dir": model, "v1": v1, "v2": v2, "prepared": prepared,
               "input_roots": {"v1": str(v1), "v2": str(v2), "prepared": str(prepared), "model": str(model)},
               "input_identity": {"synthetic_fixture": True}, "sources": sources,
               "prepared_hashes": {"synthetic_fixture": True},
               "training": train, "roles": roles, "fit_gold": fit_gold, "predict": predict}
    monkeypatch.setattr(study, "_context", lambda *args, **kwargs: deepcopy(context))
    observed_encode = []

    def artificial_encode(texts, model_dir, active_protocol):
        texts = list(texts)
        observed_encode.append(texts)
        matrix = np.zeros((len(texts), 384), dtype=np.float32)
        for i, text in enumerate(texts):
            matrix[i, int(re.search(r"classword(\d+)", text).group(1))] = 1.0
        return matrix, {"synthetic_fixture": True, "count": len(texts), "dimension": 384,
                        "encode_seconds": 0.01, "model_identity": context["model_identity"]}

    monkeypatch.setattr(encoder, "encode", artificial_encode)
    return SimpleNamespace(v1=v1, v2=v2, prepared=prepared, model=model, output=output,
                           protocol=protocol, protocol_path=protocol_path, context=context,
                           labels=labels, gold=gold, observed_encode=observed_encode)


def fit(f):
    return study.fit(f.v1, f.v2, f.prepared, f.model, f.output, protocol_path=f.protocol_path)


def evaluate(f):
    return study.evaluate(f.v1, f.v2, f.prepared, f.model, f.output, protocol_path=f.protocol_path)


def guard_gold(monkeypatch, f):
    original = Path.open
    forbidden = (f.v1 / "data/gold.json").resolve()

    def guarded(path, *args, **kwargs):
        if path.resolve() == forbidden:
            raise AssertionError("Forbidden gold read, including hashing")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)


def test_all_three_models_freeze_before_gold_and_preserve_inputs(fixture, monkeypatch):
    f = fixture
    inputs = tuple(snapshot(root) for root in (f.v1, f.v2, f.prepared, f.model))
    with monkeypatch.context() as guarded:
        guard_gold(guarded, f)
        fit(f)
    assert tuple(snapshot(root) for root in (f.v1, f.v2, f.prepared, f.model)) == inputs
    assert (f.output / "freeze.json").is_file()
    for recipe in ("tfidf1600", "minilm400", "minilm1600"):
        rows = [json.loads(line) for line in (f.output / f"{recipe}.predictions.jsonl").read_text().splitlines()]
        assert len(rows) == 1850
        assert Counter(row["split"] for row in rows) == Counter(validation=400, test=600, stress=850)
    assert not (f.output / "evaluation-access.json").exists()


def test_tfidf_fits_only_full_training_texts_and_dense_budgets_are_distinct(fixture, monkeypatch):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    original_vectorizer = TfidfVectorizer.fit_transform
    original_classifier = LogisticRegression.fit
    vectorizer_counts, classifier_counts = [], []

    def vectorizer(self, docs, *args, **kwargs):
        docs = list(docs)
        vectorizer_counts.append(len(docs))
        assert all("trainonlymarker" in doc for doc in docs)
        result = original_vectorizer(self, docs, *args, **kwargs)
        assert all(token not in self.vocabulary_ for token in
                   ("validationonlymarker", "testonlymarker", "stressonlymarker"))
        return result

    def classifier(self, matrix, labels, *args, **kwargs):
        classifier_counts.append((matrix.shape[0], Counter(labels)))
        return original_classifier(self, matrix, labels, *args, **kwargs)

    monkeypatch.setattr(TfidfVectorizer, "fit_transform", vectorizer)
    monkeypatch.setattr(LogisticRegression, "fit", classifier)
    fit(fixture)
    assert vectorizer_counts == [1600]
    assert [count for count, _ in classifier_counts] == [1600, 400, 1600]
    assert classifier_counts[1][1]["oos"] == 100
    assert all(classifier_counts[1][1][label] == 20 for label in fixture.labels[:-1])
    assert [len(rows) for rows in fixture.observed_encode] == [3450]


@pytest.mark.parametrize("artifact", ["freeze.json", "embeddings.npy", "minilm1600.predictions.jsonl",
                                      "minilm400.model.json", "tfidf1600.fit.json"])
def test_missing_or_tampered_evidence_blocks_before_gold(fixture, monkeypatch, artifact):
    fit(fixture)
    path = fixture.output / artifact
    if artifact == "freeze.json":
        path.unlink()
    else:
        path.write_bytes(path.read_bytes() + b" \n")
    with monkeypatch.context() as guarded:
        guard_gold(guarded, fixture)
        with pytest.raises((ValueError, RuntimeError, OSError)):
            evaluate(fixture)
    assert not (fixture.output / "evaluation-access.json").exists()


def test_evaluation_records_access_first_and_never_fits(fixture, monkeypatch):
    fit(fixture)
    original = Path.open
    reads = []

    def observed(path, *args, **kwargs):
        if path.resolve() == (fixture.v1 / "data/gold.json").resolve():
            assert (fixture.output / "freeze.json").is_file()
            assert (fixture.output / "evaluation-access.json").is_file()
            reads.append(path)
        return original(path, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("Evaluation attempted fitting or encoding")

    monkeypatch.setattr(Path, "open", observed)
    monkeypatch.setattr(study, "_fit_tfidf", forbidden)
    monkeypatch.setattr(study, "_fit_dense", forbidden)
    monkeypatch.setattr(encoder, "encode", forbidden)
    result = evaluate(fixture)
    assert reads and result["status"] == "COMPLETE"


def test_class_axis_and_oos_stress_denominators(fixture, monkeypatch):
    from sklearn.linear_model import LogisticRegression

    def predict(self, matrix):
        probabilities = np.zeros((matrix.shape[0], len(self.classes_)))
        probabilities[:, list(self.classes_).index("oos")] = 1.0
        return probabilities

    monkeypatch.setattr(LogisticRegression, "predict_proba", predict)
    fit(fixture)
    for recipe in fixture.protocol["recipes"]:
        rows = [json.loads(line) for line in (fixture.output / f"{recipe}.predictions.jsonl").read_text().splitlines()]
        assert all(row["predicted_label"] == "oos" and row["probabilities"][-1] == 1 for row in rows)
    result = evaluate(fixture)
    for metrics in result["metrics"].values():
        assert metrics["main_test"]["accuracy"] == 0.25
        assert metrics["main_test"]["banking_accuracy"] == 0
        assert metrics["oos_only_stress"] == {"count": 850, "oos_recall": 1.0}
        assert metrics["historical_mixed_stress"]["count"] == 1300
        assert metrics["historical_mixed_stress"]["overlapping_main_banking"] == 450
        assert metrics["historical_mixed_stress"]["accuracy"] == pytest.approx(850 / 1300)


def test_complete_reuse_and_partial_fit_preservation(fixture):
    fit(fixture)
    before = snapshot(fixture.output)
    fit(fixture)
    assert snapshot(fixture.output) == before
    (fixture.output / "freeze.json").unlink()
    partial = snapshot(fixture.output)
    with pytest.raises((ValueError, RuntimeError)):
        fit(fixture)
    assert snapshot(fixture.output) == partial


def test_partial_evaluation_blocks_another_gold_access(fixture, monkeypatch):
    fit(fixture)
    evaluate(fixture)
    (fixture.output / "evaluation.json").unlink()
    partial = snapshot(fixture.output)
    guard_gold(monkeypatch, fixture)
    with pytest.raises((ValueError, RuntimeError)):
        evaluate(fixture)
    assert snapshot(fixture.output) == partial


@pytest.mark.parametrize("source", ["v1", "v2", "prepared", "model"])
def test_output_roots_must_not_overlap_inputs(fixture, source):
    root = getattr(fixture, source)
    before = snapshot(root)
    with pytest.raises((ValueError, RuntimeError)):
        study.fit(fixture.v1, fixture.v2, fixture.prepared, fixture.model,
                  root / "output", protocol_path=fixture.protocol_path)
    assert snapshot(root) == before


def test_active_lock_preserved(fixture):
    fixture.output.mkdir()
    lock = fixture.output / ".baseline.lock"
    lock.write_text("Artificial owner\n")
    with pytest.raises((ValueError, RuntimeError)):
        fit(fixture)
    assert lock.read_text() == "Artificial owner\n"


def test_export_validates_freeze_again(fixture):
    fit(fixture)
    evaluate(fixture)
    path = fixture.output / "minilm400.model.json"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises((ValueError, RuntimeError)):
        study.bundle(fixture.output)


def test_complete_bundle_reuses_original_bytes(fixture):
    import zipfile
    fit(fixture)
    evaluate(fixture)
    bundle = study.bundle(fixture.output)
    before = bundle.read_bytes()
    assert study.bundle(fixture.output).read_bytes() == before
    with zipfile.ZipFile(bundle) as archive:
        assert len(archive.namelist()) == len(set(archive.namelist()))
        for name in archive.namelist():
            assert archive.read(name) == (fixture.output / name).read_bytes()


def test_nonfinite_embeddings_never_freeze(fixture, monkeypatch):
    def invalid(texts, *args, **kwargs):
        matrix = np.zeros((len(texts), 384), dtype=np.float32)
        matrix[0, 0] = np.nan
        return matrix, {"synthetic_fixture": True}
    monkeypatch.setattr(encoder, "encode", invalid)
    with pytest.raises((ValueError, RuntimeError)):
        fit(fixture)
    assert not (fixture.output / "freeze.json").exists()


def test_mean_pool_masks_padding_and_normalizes():
    tokens = np.array([[[1., 0.], [0., 1.], [100., 100.]]], dtype=np.float32)
    mask = np.array([[1, 1, 0]], dtype=np.int64)
    result = encoder.mean_pool(tokens, mask)
    assert result.shape == (1, 2)
    assert np.allclose(result, [[2 ** -.5, 2 ** -.5]], atol=1e-7)


@pytest.mark.parametrize("failure", ["nan", "zero-mask", "zero-norm"])
def test_mean_pool_rejects_invalid_values(failure):
    tokens = np.ones((1, 2, 2), dtype=np.float32)
    mask = np.ones((1, 2), dtype=np.int64)
    if failure == "nan":
        tokens[0, 0, 0] = np.nan
    elif failure == "zero-mask":
        mask[:] = 0
    else:
        tokens[:] = 0
    with pytest.raises((ValueError, RuntimeError)):
        encoder.mean_pool(tokens, mask)


@pytest.fixture
def data_fixture(fixture, monkeypatch):
    """Construct fake official-shaped sources, then exercise real data APIs."""
    f = fixture
    raw = f.v1.parent / "raw"
    raw.mkdir()
    f.raw = raw
    full = f.context["training"]["1600"]
    development = []
    for label in f.labels[:-1]:
        selected = [row for row in full if row["label"] == label and
                    int(row["sample_id"].split(":")[2]) % 100 in (20, 21, 22, 23)]
        development.extend({k: v for k, v in row.items() if k != "label"} |
                           {"task": "clinc", "split": "development", "labels": f.labels}
                           for row in selected)
    development += [{"sample_id": f"clinc:authored-oos:{i}", "state": f"ARTIFICIAL authored domain {i}",
                     "task": "clinc", "split": "development", "labels": f.labels}
                    for i in range(4)]
    allrows = sum(f.context["roles"].values(), []) + development
    def rows(path, values):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in values))
    f.write_rows = rows
    rows(f.v1 / "data/examples.jsonl", allrows)
    rows(f.v2 / "data/examples.jsonl", f.context["roles"]["calibration"] + f.context["roles"]["validation"])
    dump(f.v2 / "data/fit_gold.json", f.context["fit_gold"])
    dump(f.v1 / "data/manifest.json", {"protocol": "solo-v1", "files": {
        "examples.jsonl": sha(f.v1 / "data/examples.jsonl"), "gold.json": sha(f.v1 / "data/gold.json")}})
    sources = {"v1_manifest_sha256": f.v1 / "data/manifest.json", "v1_examples_sha256": f.v1 / "data/examples.jsonl",
               "v1_gold_sha256": f.v1 / "data/gold.json", "v2_examples_sha256": f.v2 / "data/examples.jsonl",
               "v2_fit_gold_sha256": f.v2 / "data/fit_gold.json"}
    monkeypatch.setattr(data, "SOURCE_PINS", {name: sha(path) for name, path in sources.items()})
    by_source = {source: sorted((row for row in full if row["source_split"] == source),
                                key=lambda row: int(row["sample_id"].split(":")[2]))
                 for source in ("train", "oos_train")}
    raw_json = {source: [[row["state"], row["label"]] for row in values]
                for source, values in by_source.items()}
    raw_json["train"].extend([[f"ARTIFICIAL ignored class example {i}", "non_banking_intent"]
                              for i in range(13500)])
    raw_json["test"] = [["ARTIFICIAL source test label must never enter fitting", "oos"]]
    dump(raw / "data_full.json", raw_json)
    dump(raw / "domains.json", {"banking": f.labels[:-1]})
    monkeypatch.setattr(data, "RAW_PINS", {name: sha(raw / name) for name in ("data_full.json", "domains.json")})
    contents = data._prepared_content(raw, data._project(f.v1, f.v2))
    monkeypatch.setattr(data, "PREPARED_PINS", {name: hashlib.sha256(content).hexdigest()
                                               for name, content in contents.items()})
    (f.prepared / "unchanged.txt").unlink()
    f.prepared_contents = contents
    return f


def test_prepare_and_load_never_open_project_gold_and_load_never_opens_raw(data_fixture, monkeypatch):
    f = data_fixture
    with monkeypatch.context() as guarded:
        guard_gold(guarded, f)
        result = data.prepare(f.raw, f.v1, f.v2, f.prepared)
    assert result["training_count"] == 1600
    assert result["audit"]["engineering_development_included"] == 60
    original = Path.open

    def guarded(path, *args, **kwargs):
        resolved = path.resolve()
        if resolved == (f.v1 / "data/gold.json").resolve() or f.raw.resolve() in resolved.parents:
            raise AssertionError("Fitting input loader opened raw all-splits data or V1 gold")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    context = data.load(f.prepared, f.v1, f.v2)
    assert {budget: len(values) for budget, values in context["training"].items()} == {"400": 400, "1600": 1600}
    assert len(context["predict"]) == 1850
    assert Counter(row["label"] for row in context["training"]["1600"]) == Counter({label: 100 for label in f.labels})


def test_prepared_data_conflicts_are_not_replaced(data_fixture):
    f = data_fixture
    data.prepare(f.raw, f.v1, f.v2, f.prepared)
    before = snapshot(f.prepared)
    data.prepare(f.raw, f.v1, f.v2, f.prepared)
    assert snapshot(f.prepared) == before
    path = f.prepared / "train1600.jsonl"
    path.write_bytes(path.read_bytes() + b"\n")
    conflict = snapshot(f.prepared)
    with pytest.raises(data.DataError):
        data.prepare(f.raw, f.v1, f.v2, f.prepared)
    assert snapshot(f.prepared) == conflict


def test_training_population_validation_survives_updated_file_hashes(data_fixture, monkeypatch):
    f = data_fixture
    data.prepare(f.raw, f.v1, f.v2, f.prepared)
    path = f.prepared / "train1600.jsonl"
    values = [json.loads(line) for line in path.read_text().splitlines()]
    values[0]["label"] = "oos" if values[0]["label"] != "oos" else "balance"
    f.write_rows(path, values)
    manifest = json.loads((f.prepared / "manifest.json").read_text())
    manifest["files"]["train1600.jsonl"] = sha(path)
    dump(f.prepared / "manifest.json", manifest)
    monkeypatch.setattr(data, "PREPARED_PINS", {name: sha(f.prepared / name)
                                               for name in ("train1600.jsonl", "manifest.json")})
    guard_gold(monkeypatch, f)
    with pytest.raises(data.DataError):
        data.load(f.prepared, f.v1, f.v2)


@pytest.fixture
def encoder_files(tmp_path, monkeypatch):
    root = tmp_path / "fixed-encoder"
    contents = {
        "onnx/model.onnx": b"Artificial encoder binary; not a real model\n",
        "tokenizer.json": b"{}\n", "tokenizer_config.json": b"{}\n", "special_tokens_map.json": b"{}\n",
        "config.json": json.dumps({"hidden_size": 384, "pad_token_id": 0}).encode(),
        "sentence_bert_config.json": json.dumps({"max_seq_length": 256}).encode(),
        "1_Pooling/config.json": json.dumps({"word_embedding_dimension": 384, "pooling_mode_cls_token": False,
            "pooling_mode_mean_tokens": True, "pooling_mode_max_tokens": False,
            "pooling_mode_mean_sqrt_len_tokens": False}).encode(),
        "modules.json": json.dumps([{"type": f"sentence_transformers.models.{name}"}
                                    for name in ("Transformer", "Pooling", "Normalize")]).encode()}
    spec = deepcopy(encoder.ENCODER_SPEC)
    spec["files"] = {}
    for name, content in contents.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        spec["files"][name] = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    monkeypatch.setattr(encoder, "ENCODER_SPEC", spec)
    return SimpleNamespace(root=root, protocol={"encoder": spec})


def test_encoder_file_identity_is_checked_before_loading(encoder_files, monkeypatch):
    f = encoder_files
    encoder.verify(f.root, f.protocol)
    path = f.root / "onnx/model.onnx"
    path.write_bytes(path.read_bytes() + b"tampered")

    def forbidden(*args, **kwargs):
        raise AssertionError("Loading happened before identity verification")

    monkeypatch.setattr(encoder, "_load_tokenizer", forbidden)
    monkeypatch.setattr(encoder, "_load_session", forbidden)
    with pytest.raises(encoder.EncoderError, match="changed"):
        encoder.encode(["ARTIFICIAL input"], f.root, f.protocol)


def test_encoder_semantic_config_is_checked_beyond_file_hashes(encoder_files):
    f = encoder_files
    path = f.root / "1_Pooling/config.json"
    pooling = json.loads(path.read_text())
    pooling["pooling_mode_cls_token"] = True
    dump(path, pooling)
    f.protocol["encoder"]["files"]["1_Pooling/config.json"] = {
        "bytes": path.stat().st_size, "sha256": sha(path)}
    with pytest.raises(encoder.EncoderError, match="configuration"):
        encoder.verify(f.root, f.protocol)


def fake_encoder_runtime(monkeypatch, lengths, *, overflow=False):
    tokens = [SimpleNamespace(ids=[1] * length, type_ids=[0] * length,
                              attention_mask=[1] * length, overflowing=[object()] if overflow else [])
              for length in lengths]
    calls = []

    class Tokenizer:
        def encode_batch(self, texts, add_special_tokens):
            assert add_special_tokens is True
            assert len(texts) == len(tokens)
            return tokens

    class Session:
        def get_inputs(self):
            return [SimpleNamespace(name=name, type="tensor(int64)")
                    for name in ("input_ids", "attention_mask", "token_type_ids")]
        def get_outputs(self):
            return [SimpleNamespace(name="last_hidden_state")]
        def run(self, outputs, inputs):
            calls.append(inputs)
            assert outputs == ["last_hidden_state"]
            assert all(matrix.dtype == np.int64 for matrix in inputs.values())
            batch, width = inputs["input_ids"].shape
            result = np.ones((batch, width, 384), dtype=np.float32)
            result[inputs["attention_mask"] == 0] = 1000.0
            return [result]

    monkeypatch.setattr(encoder, "verify", lambda *args: {"synthetic_fixture": True})
    monkeypatch.setattr(encoder, "_load_tokenizer", lambda path: Tokenizer())
    monkeypatch.setattr(encoder, "_load_session", lambda *args: Session())
    return calls


@pytest.mark.parametrize("length,overflow", [(257, False), (4, True)])
def test_encoder_rejects_truncation_before_session_creation(tmp_path, monkeypatch, length, overflow):
    fake_encoder_runtime(monkeypatch, [length], overflow=overflow)

    def forbidden(*args, **kwargs):
        raise AssertionError("An overlength or truncated input reached model startup")

    monkeypatch.setattr(encoder, "_load_session", forbidden)
    with pytest.raises(encoder.EncoderError, match="truncation"):
        encoder.encode(["ARTIFICIAL input"], tmp_path, {"encoder": deepcopy(encoder.ENCODER_SPEC)})


def test_encoder_exact_limit_and_padding_keep_complete_rows(tmp_path, monkeypatch):
    calls = fake_encoder_runtime(monkeypatch, [256, 3])
    matrix, meta = encoder.encode(["ARTIFICIAL long", "ARTIFICIAL short"], tmp_path,
                                  {"encoder": deepcopy(encoder.ENCODER_SPEC)})
    assert matrix.shape == (2, 384) and matrix.dtype == np.float32
    assert np.allclose(matrix[0], matrix[1], rtol=0, atol=1e-7)
    assert meta["token_lengths"] == [256, 3]
    assert meta["truncated_count"] == 0
    assert calls[0]["attention_mask"][1].sum() == 3
    assert np.all(calls[0]["input_ids"][1, 3:] == 0)


def reseal_freeze(f, transform):
    path = f.output / "freeze.json"
    freeze = json.loads(path.read_text())
    freeze.pop("self_hash")
    transform(freeze)
    path.write_bytes(study._bytes(study._seal(freeze)))


def test_freeze_must_bind_every_recipe_even_if_unbound_files_still_exist(fixture, monkeypatch):
    fit(fixture)
    def omit_third(freeze):
        freeze["artifacts"] = {name: value for name, value in freeze["artifacts"].items()
                               if not name.startswith("minilm1600.")}
    reseal_freeze(fixture, omit_third)
    guard_gold(monkeypatch, fixture)
    with pytest.raises(study.BaselineError, match="three-recipe"):
        evaluate(fixture)


def test_feature_input_identity_is_checked_beyond_artifact_hashes(fixture, monkeypatch):
    fit(fixture)
    path = fixture.output / "embeddings.json"
    value = json.loads(path.read_text())
    value["inputs"][0], value["inputs"][1] = value["inputs"][1], value["inputs"][0]
    dump(path, value)
    reseal_freeze(fixture, lambda freeze: freeze["artifacts"].update({"embeddings.json": sha(path)}))
    guard_gold(monkeypatch, fixture)
    with pytest.raises(study.BaselineError, match="feature identity"):
        evaluate(fixture)


def test_prediction_ids_are_checked_beyond_artifact_hashes(fixture, monkeypatch):
    fit(fixture)
    path = fixture.output / "minilm400.predictions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["sample_id"] = rows[1]["sample_id"]
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    reseal_freeze(fixture, lambda freeze: freeze["artifacts"].update({path.name: sha(path)}))
    guard_gold(monkeypatch, fixture)
    with pytest.raises(study.BaselineError, match="ID/order/class"):
        evaluate(fixture)


def test_cli_run_and_export_reuse(fixture, monkeypatch, capsys):
    monkeypatch.setattr(study, "_source_identity", lambda: {"git_available": True, "git_dirty": False})
    arguments = ["run", "--v1-dir", str(fixture.v1), "--v2-dir", str(fixture.v2),
                 "--prepared-dir", str(fixture.prepared), "--model-dir", str(fixture.model),
                 "--output-dir", str(fixture.output)]
    assert study.main(arguments) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "COMPLETE"
    assert set(result["main_test"]) == {"tfidf1600", "minilm400", "minilm1600"}
    assert all(value["count"] == 600 for value in result["main_test"].values())
    bundle = Path(result["bundle"])
    original = bundle.read_bytes()
    assert study.main(arguments) == 0
    assert bundle.read_bytes() == original


@pytest.mark.parametrize("identity", [{"git_available": False, "git_dirty": None},
                                      {"git_available": True, "git_dirty": True}])
def test_cli_requires_clean_versioned_source(fixture, monkeypatch, identity):
    monkeypatch.setattr(study, "_source_identity", lambda: identity)
    assert study.main(["bundle", "--output-dir", str(fixture.output)]) == 2
    assert not fixture.output.exists()


def test_real_context_and_pipeline_keep_raw_and_gold_out_of_fit(data_fixture, monkeypatch):
    """Real input loader + real LR pipeline; only encoder output is artificial."""
    f = data_fixture
    data.prepare(f.raw, f.v1, f.v2, f.prepared)
    f.protocol["sources"] = dict(data.SOURCE_PINS)
    f.protocol["prepared"] = dict(data.PREPARED_PINS)
    dump(f.protocol_path, f.protocol)
    monkeypatch.setattr(study, "_context", REAL_CONTEXT)
    monkeypatch.setattr(encoder, "verify", lambda *args: {"synthetic_fixture": True})
    # This test may also run in an older optional CPU environment. Dependency
    # installation is validated separately; keep the real data/context logic.
    monkeypatch.setattr(study, "environment_identity", lambda protocol: {
        "packages": protocol["environment"], "synthetic_fixture": True})
    original = Path.open

    def guarded(path, *args, **kwargs):
        resolved = path.resolve()
        if resolved == (f.v1 / "data/gold.json").resolve() or f.raw.resolve() in resolved.parents:
            raise AssertionError("Fit accessed raw all-splits data or V1 gold")
        return original(path, *args, **kwargs)

    with monkeypatch.context() as protected:
        protected.setattr(Path, "open", guarded)
        fit(f)
    freeze = json.loads((f.output / "freeze.json").read_text())
    assert freeze["sources"] == data.SOURCE_PINS
    assert freeze["prepared_hashes"] == data.PREPARED_PINS
    assert {name: len(values) for name, values in freeze["training_ids"].items()} == {
        "tfidf1600": 1600, "minilm400": 400, "minilm1600": 1600}
