"""Read-only archive, release and input identity audit for clinc-cheap-v1."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import subprocess
import zipfile


def sha(value):
    return hashlib.sha256(value).hexdigest()


def canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "repo", "artifact-root", "prepared-dir", "model-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    expected_zip = "7f00178c7bd1d86eeeb121929131447675d0053c10072f687f88d7e3ae77b2be"
    commit = "1b70899bbcf15ee686df04b36e0c333a2fc21e5c"
    recipes = ("tfidf1600", "minilm400", "minilm1600")
    fitted = {"protocol.json", "embeddings.npy", "embeddings.json"} | {
        f"{recipe}.{suffix}" for recipe in recipes
        for suffix in ("model.json", "predictions.jsonl", "fit.json")}
    names = fitted | {"freeze.json", "evaluation-access.json", "evaluation.json"}
    archive_bytes = args.bundle.read_bytes()
    assert sha(archive_bytes) == expected_zip
    with zipfile.ZipFile(args.bundle) as archive:
        assert len(archive.namelist()) == len(names) and set(archive.namelist()) == names
        assert sum(i.file_size for i in archive.infolist()) < 30_000_000
        for info in archive.infolist():
            assert not PurePosixPath(info.filename).is_absolute()
            assert ".." not in PurePosixPath(info.filename).parts
            assert not stat.S_ISLNK(info.external_attr >> 16) and not info.flag_bits & 1
        assert archive.testzip() is None
        contents = {name: archive.read(name) for name in names}
    parsed = {n: json.loads(b) for n, b in contents.items() if n.endswith(".json")}
    freeze, access, evaluation = [parsed[n] for n in ("freeze.json", "evaluation-access.json", "evaluation.json")]
    for value in (freeze, access, evaluation):
        assert value["self_hash"] == sha(canonical({k: v for k, v in value.items() if k != "self_hash"}))
    assert freeze["status"] == "FROZEN" and evaluation["status"] == "COMPLETE"
    assert set(freeze["artifacts"]) == fitted
    for name, expected in freeze["artifacts"].items():
        assert sha(contents[name]) == expected
    for value in (access, evaluation):
        assert value["freeze_sha256"] == sha(contents["freeze.json"])
        assert value["gold_sha256"] == freeze["sources"]["v1_gold_sha256"]
    assert evaluation["access_sha256"] == sha(contents["evaluation-access.json"])
    assert set(evaluation["metrics"]) == set(recipes)
    source = freeze["source_identity"]
    assert source["git_commit"] == commit and source["git_available"] is True and source["git_dirty"] is False
    expected_source_files = {f"decision_study/cheap_baselines/{n}" for n in
                             ("__init__.py", "__main__.py", "data.py", "encoder.py", "study.py")}
    expected_source_files |= {"configs/cheap_baseline_requirements.txt", "scripts/run_cheap_baselines.sh",
                              "scripts/run_cheap_baseline_suite.sh"}
    assert set(source["files"]) == expected_source_files
    def released(name):
        return subprocess.run(["git", "-C", str(args.repo), "show", f"{commit}:{name}"],
                              capture_output=True, check=True, timeout=15).stdout
    for name, expected in source["files"].items():
        assert sha(released(name)) == expected
        assert sha((args.repo / name).read_bytes()) == expected
    protocol_raw = released("configs/clinc_cheap_protocol.json")
    assert sha(protocol_raw) == freeze["protocol_sha256"]
    protocol = parsed["protocol.json"]
    assert json.loads(protocol_raw) == protocol
    assert freeze["sources"] == protocol["sources"]
    assert freeze["prepared_hashes"] == protocol["prepared"]
    assert freeze["environment"]["packages"] == protocol["environment"]
    assert freeze["environment"]["threads"] == 1
    assert freeze["input_roots"] == {
        "v1": "/home/zhanj289/e6998/decision-study/results/solo-v1",
        "v2": "/home/zhanj289/e6998/decision-study-src/results/solo-v2",
        "prepared": "/home/zhanj289/e6998/decision-study/results/aux-clinc-cheap-v1-inputs/prepared",
        "model": "/home/zhanj289/e6998/decision-study/results/aux-clinc-cheap-v1-inputs/model"}
    v1, v2 = "results/returned-experiment-2026-11-02-101054", "results/solo-v2-prepared"
    sources = {"v1_manifest_sha256": f"{v1}/data/manifest.json", "v1_examples_sha256": f"{v1}/data/examples.jsonl",
               "v1_gold_sha256": f"{v1}/data/gold.json", "v2_examples_sha256": f"{v2}/data/examples.jsonl",
               "v2_fit_gold_sha256": f"{v2}/data/fit_gold.json"}
    # Historical labels are checked only after the completed access chain above.
    for key, name in sources.items():
        assert sha((args.artifact_root / name).read_bytes()) == protocol["sources"][key]
    for name, expected in protocol["prepared"].items():
        assert sha((args.prepared_dir / name).read_bytes()) == expected
    spec = protocol["encoder"]
    identity = {key: spec[key] for key in ("repository", "revision", "files", "provider")}
    assert freeze["model_identity"] == identity
    for name, expected in spec["files"].items():
        value = (args.model_dir / name).read_bytes()
        assert len(value) == expected["bytes"] and sha(value) == expected["sha256"]
    features = parsed["embeddings.json"]
    assert features["model_identity"] == features["encoding"]["identity"] == identity
    assert features["matrix_sha256"] == sha(contents["embeddings.npy"])
    assert features["encoding"]["count"] == len(features["inputs"]) == 3450
    assert features["encoding"]["truncated_count"] == 0
    assert evaluation["interpretation"] == protocol["interpretation"]
    for recipe in recipes:
        fit = parsed[recipe + ".fit.json"]
        assert fit["recipe"] == recipe and fit["prediction_count"] == 1850
        assert fit["training_count"] == (400 if recipe == "minilm400" else 1600)
        assert fit["model_bytes"] == len(contents[recipe + ".model.json"])
        assert fit["interpretation"] == protocol["interpretation"]
    receipt = {
        "status": "PASS", "archive_sha256": expected_zip, "archive_bytes": len(archive_bytes),
        "members": {n: {"bytes": len(b), "sha256": sha(b)} for n, b in sorted(contents.items())},
        "source_identity": source, "protocol_source_sha256": sha(protocol_raw),
        "protocol_saved_sha256": sha(contents["protocol.json"]), "protocol_parsed_content_equal": True,
        "input_sources": {key: {"relative_path": name, "sha256": protocol["sources"][key]} for key, name in sources.items()},
        "prepared_hashes": protocol["prepared"], "model_identity": identity,
        "reported_environment": freeze["environment"], "recorded_node_input_roots": freeze["input_roots"],
        "evaluation_access_at_utc": access["started_at_utc"],
        "checks": ["exact15 safe unique ZIP members and CRC", "three self hashes", "twelve frozen artifact hashes",
                   "complete freeze-access-evaluation chain", "released commit/eight sources/protocol identity",
                   "five project and two prepared inputs", "eight actual encoder files",
                   "24 dependency versions", "encoder metadata identities", "fit metadata consistency"],
        "limits": ["Local hash chains are integrity evidence, not signatures or OS file-access traces.",
                   "Original Linux elapsed durations and hardware were not independently remeasured.",
                   "This script performs no classifier fitting or model service calls."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical(receipt))
    print(json.dumps({"status": "PASS", "archive_members": len(names), "source_commit": commit}))


if __name__ == "__main__":
    main()
