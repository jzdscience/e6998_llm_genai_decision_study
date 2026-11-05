"""Read-only audit of the returned CLINC400 archive against its released inputs."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import subprocess
import zipfile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2,
                       allow_nan=False) + "\n").encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    archive_bytes = args.bundle.read_bytes()
    expected_sha = "7fcd28785a06bf235ee46ca829e4d839cf88743894c5b965cc3e04fcb215414f"
    assert digest(archive_bytes) == expected_sha
    names = {"protocol.json", "model.json", "predictions.jsonl", "fit.json",
             "freeze.json", "evaluation-access.json", "evaluation.json"}
    with zipfile.ZipFile(args.bundle) as archive:
        infos = archive.infolist()
        assert len(infos) == len(names) and set(archive.namelist()) == names
        assert sum(item.file_size for item in infos) < 10_000_000
        for item in infos:
            assert not PurePosixPath(item.filename).is_absolute()
            assert ".." not in PurePosixPath(item.filename).parts
            assert not stat.S_ISLNK(item.external_attr >> 16)
            assert not item.flag_bits & 1
        contents = {name: archive.read(name) for name in names}
        assert archive.testzip() is None
    objects = {name: json.loads(data) for name, data in contents.items()
               if name.endswith(".json")}
    freeze, access, evaluation = [objects[name] for name in
                                  ("freeze.json", "evaluation-access.json", "evaluation.json")]
    for value in (freeze, access, evaluation):
        assert value["self_hash"] == digest(canonical({k: v for k, v in value.items()
                                                      if k != "self_hash"}))
    assert freeze["status"] == "FROZEN" and evaluation["status"] == "COMPLETE"
    assert set(freeze["artifacts"]) == names - {"freeze.json", "evaluation-access.json", "evaluation.json"}
    for name, expected in freeze["artifacts"].items():
        assert digest(contents[name]) == expected
    for value in (access, evaluation):
        assert value["freeze_sha256"] == digest(contents["freeze.json"])
        assert value["gold_sha256"] == freeze["sources"]["v1_gold_sha256"]
    assert evaluation["access_sha256"] == digest(contents["evaluation-access.json"])
    source = freeze["source_identity"]
    commit = "797aff1edfe17d7daaf93551e6cfc236035b6945"
    assert source["git_commit"] == commit and source["git_available"] is True
    assert source["git_dirty"] is False
    expected_sources = {f"decision_study/baselines/{name}" for name in
                        ("__init__.py", "__main__.py", "study.py")}
    assert set(source["files"]) == expected_sources
    def released(name):
        return subprocess.run(["git", "-C", str(args.repo), "show", f"{commit}:{name}"],
                              capture_output=True, check=True, timeout=10).stdout
    for name, expected in source["files"].items():
        assert digest(released(name)) == expected
        assert digest((args.repo / name).read_bytes()) == expected
    protocol_bytes = released("configs/clinc_tfidf400_protocol.json")
    assert digest(protocol_bytes) == freeze["protocol_sha256"]
    assert json.loads(protocol_bytes) == objects["protocol.json"]
    protocol = objects["protocol.json"]
    assert freeze["sources"] == protocol["sources"]
    assert freeze["label_order"] == protocol["labels"]
    assert freeze["environment"]["packages"] == protocol["environment"]
    assert freeze["environment"]["threads"] == 1
    assert freeze["input_roots"] == {
        "v1": "/home/zhanj289/e6998/decision-study/results/solo-v1",
        "v2": "/home/zhanj289/e6998/decision-study-src/results/solo-v2"}
    v1 = "results/returned-experiment-2026-11-02-101054"
    v2 = "results/solo-v2-prepared"
    inputs = {"v1_manifest_sha256": f"{v1}/data/manifest.json",
              "v1_examples_sha256": f"{v1}/data/examples.jsonl",
              "v1_gold_sha256": f"{v1}/data/gold.json",
              "v2_examples_sha256": f"{v2}/data/examples.jsonl",
              "v2_fit_gold_sha256": f"{v2}/data/fit_gold.json"}
    # This is a retrospective audit after verifying COMPLETE and the access chain.
    for key, name in inputs.items():
        assert digest((args.artifact_root / name).read_bytes()) == freeze["sources"][key]
    fit = objects["fit.json"]
    assert fit["model_bytes"] == len(contents["model.json"])
    assert fit["feature_count"] == len(objects["model.json"]["vocabulary"])
    assert fit["training_count"] == len(freeze["training_ids"]) == 400
    assert fit["interpretation"] == evaluation["interpretation"] == protocol["interpretation"]
    receipt = {
        "status": "PASS", "archive_sha256": expected_sha, "archive_bytes": len(archive_bytes),
        "verified_members": {name: {"bytes": len(data), "sha256": digest(data)}
                             for name, data in sorted(contents.items())},
        "source_identity": source, "source_protocol_file_sha256": digest(protocol_bytes),
        "saved_protocol_canonical_sha256": digest(contents["protocol.json"]),
        "protocol_note": "The source config and saved canonical JSON have different bytes but identical parsed content.",
        "input_sources": {key: {"path_relative_to_external_artifact_root": name,
                                "sha256": freeze["sources"][key]} for key, name in inputs.items()},
        "reported_environment": freeze["environment"], "recorded_node_input_roots": freeze["input_roots"],
        "evaluation_access_at_utc": access["started_at_utc"],
        "checks": ["exact safe unique ZIP inventory and CRC", "three self hashes",
                   "four frozen artifact hashes", "freeze-access-evaluation links",
                   "released commit and three source files", "raw source protocol and canonical saved protocol",
                   "five historical input hashes", "pinned package versions and node roots",
                   "fit size/count and interpretation consistency"],
        "limits": ["Hash chains are integrity evidence, not signatures or independent OS file-access traces.",
                   "Recorded Linux environment and elapsed times were not remeasured remotely.",
                   "This read-only audit performs no fitting, evaluation command, or model-service call."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical(receipt))
    print(json.dumps({"status": receipt["status"], "members": len(names),
                      "source_commit": commit, "output": str(args.output)}))


if __name__ == "__main__":
    main()
