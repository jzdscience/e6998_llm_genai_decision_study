"""Portable, checksummed result bundles. Never include weights, environments, or credentials."""
from __future__ import annotations

import hashlib
import json
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from .common import StudyError, file_hash, fingerprint, read_json, read_jsonl, utc_now, write_json
from .storage import load_predictions


def bundle_readiness(output_dir: Path) -> dict:
    from .data import load_examples, load_gold, verify_data
    root = Path(output_dir)
    missing = []
    try:
        verify_data(root)
        examples = load_examples(root)
    except (ValueError, OSError, KeyError) as exc:
        return {"experiment_ready": False, "benchmark_ready": False, "missing": [f"data: {exc}"]}
    predictions, calibrations, evidence = {}, {}, set()
    for model in ("kev", "startlux", "qwen"):
        splits = {"test", "stress"} if model == "qwen" else {"calibration", "validation", "test", "stress"}
        expected = {e["sample_id"] for e in examples if e["split"] in splits}
        try:
            predictions[model] = load_predictions(root, model)
            actual = {r["sample_id"] for r in predictions[model]}
            evidence.update(bool(r.get("is_test_fixture",False)) for r in predictions[model])
            if actual != expected:
                missing.append(f"{model}: {len(expected-actual)} missing and {len(actual-expected)} unexpected terminal predictions")
        except (ValueError, OSError, StudyError) as exc:
            missing.append(f"{model}: {exc}")
        if model != "qwen":
            artifact_path = root / "calibration" / f"{model}.json"
            if not artifact_path.exists():
                missing.append(f"{model}: frozen calibration missing")
            else:
                try:
                    from .analysis import validate_calibration
                    calibrations[model] = read_json(artifact_path)
                    validate_calibration(examples,load_gold(root),predictions.get(model,[]),calibrations[model],model)
                except (ValueError,KeyError,OSError) as exc:
                    missing.append(f"{model}: invalid calibration: {exc}")
    if len(evidence) > 1:
        missing.append("Mixed artificial test predictions and model predictions")
    benchmark_missing = _benchmark_missing(root,calibrations) if not missing else ["Formal predictions/calibration incomplete"]
    return {"experiment_ready": not missing, "benchmark_ready": not benchmark_missing, "missing": missing,
            "benchmark_missing":benchmark_missing,
            "evidence_type":"software_test_fixture" if evidence=={True} else "model_predictions" if evidence else "no_predictions"}


def _benchmark_missing(root: Path, calibrations: dict) -> list[str]:
    directory = root/"benchmark"
    if not (directory/"summary.json").exists():
        return ["No completed live benchmark summary"]
    try:
        settings = read_json(directory/"settings.json")
        load = read_json(directory/"load_profile.json")
        rows = read_json(directory/"summary.json")
        if settings["benchmark_source_hash"] != file_hash(Path(__file__).with_name("benchmark.py")):
            raise StudyError("Benchmark source differs from measured implementation")
        if load["settings_hash"] != fingerprint(settings):
            raise StudyError("Benchmark load profile differs from frozen settings")
        for model in ("kev","startlux","qwen"):
            if settings["config_hashes"][model] != fingerprint(read_json(root/"predictions"/model/"config.json")):
                raise StudyError("Benchmark/prediction runtime mismatch")
        if settings["calibration_hashes"] != {m:fingerprint(c) for m,c in calibrations.items()}:
            raise StudyError("Benchmark calibration mismatch")
        deployments={"kev_direct","startlux_direct","qwen_direct","kev_to_qwen","startlux_to_qwen"}
        expected={(t,d,c,r) for t in ("clinc","squad") for d in deployments for c in {1,load["concurrency"]} for r in range(settings["repeats"])}
        actual={(r["task"],r["deployment"],r["concurrency"],r["repeat"]) for r in rows}
        if actual != expected or len(rows) != len(expected):
            raise StudyError("Missing/duplicate benchmark cells")
        for row in rows:
            if row["settings_hash"] != fingerprint(settings):
                raise StudyError("Mismatched benchmark cell")
            key=f"{row['task']}-{row['deployment']}-c{row['concurrency']}-r{row['repeat']}"
            if read_json(directory/f"{key}.json") != row:
                raise StudyError("Benchmark cell differs from summary")
            trace = read_jsonl(directory/f"{key}.trace.jsonl")
            if len(trace) != row["inputs"] or row["inputs"] < 200 or row["unique_inputs"] != 200:
                raise StudyError("Incomplete live trace")
            if row["elapsed_seconds"] < settings["minimum_round_seconds"]:
                raise StudyError("Incomplete timing interval")
    except (KeyError,ValueError,OSError,StudyError,TypeError) as exc:
        return [str(exc)]
    return []


def export_bundle(output_dir: Path, kind: str = "experiment", destination: Path | None = None) -> Path:
    root = Path(output_dir).resolve()
    if kind not in {"pilot", "experiment", "benchmark"}:
        raise ValueError(kind)
    root.mkdir(parents=True, exist_ok=True)
    readiness = bundle_readiness(root)
    status_path = root / "status.json"
    status = read_json(status_path) if status_path.exists() else {"state":"partial","stages":{}}
    required = readiness["experiment_ready"] and (kind != "benchmark" or readiness["benchmark_ready"])
    if kind == "pilot":
        reports = [root / "smoke" / f"{m}.json" for m in ("kev","startlux","qwen")]
        required = all(p.exists() and read_json(p).get("passed") for p in reports)
        if required:
            try:
                from .runner import protocol_config, require_smoke
                for model in ("kev","startlux","qwen"):
                    require_smoke(root,model,protocol_config(root,model))
            except (StudyError,ValueError,KeyError,OSError,RuntimeError):
                required = False
    status["bundle_kind"], status["bundle_complete"] = kind, required
    status["evidence_type"] = readiness.get("evidence_type","no_predictions")
    status["model_results_complete"] = required and status["evidence_type"] == "model_predictions"
    if required and status.get("state") != "blocked":
        status["state"] = "complete"
    elif status.get("state") != "blocked":
        status["state"] = "partial"
    status["readiness"] = readiness
    write_json(status_path, status)
    allowed = {"data","predictions","calibration","smoke","analysis","benchmark","runtime","diagnostics"}
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root)
        if relative.parts[0] in allowed or relative.as_posix() in {"status.json","budget.json","README_RESULTS.md"}:
            # Runtime installs and weights must live elsewhere; allow only bounded metadata/log formats.
            if relative.parts[0] == "runtime":
                if path.suffix not in {".json",".jsonl",".log",".txt"}:
                    continue
                if len(relative.parts) > 2 and relative.parts[1] not in {"logs","resolved"}:
                    continue
            files.append(path)
    manifest = {"schema_version":1,"kind":kind,"created_at":utc_now(),"complete":required,
                "evidence_type":status["evidence_type"],"model_results_complete":status["model_results_complete"],
                "files":{p.relative_to(root).as_posix():file_hash(p) for p in files}}
    target = Path(destination) if destination else root / "exports" / f"{kind}_bundle.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in files:
            archive.write(path,path.relative_to(root).as_posix())
        archive.writestr("bundle_manifest.json",json.dumps(manifest,ensure_ascii=False,indent=2))
    tmp.replace(target)
    return target


def import_bundle(zip_path: Path, output_dir: Path) -> dict:
    target = Path(output_dir).resolve()
    with zipfile.ZipFile(zip_path) as archive:
        infos = archive.infolist()
        if len({i.filename for i in infos}) != len(infos):
            raise StudyError("Duplicate ZIP member")
        if sum(i.file_size for i in infos) > 2_000_000_000:
            raise StudyError("Result bundle exceeds 2 GB; inspect before extracting")
        for entry in infos:
            name = PurePosixPath(entry.filename)
            if name.is_absolute() or ".." in name.parts or "\\" in entry.filename or ":" in entry.filename or stat.S_ISLNK(entry.external_attr >> 16):
                raise StudyError("Unsafe ZIP member")
        if "bundle_manifest.json" not in archive.namelist():
            raise StudyError("Not a checksummed decision-study result bundle")
        manifest = json.loads(archive.read("bundle_manifest.json"))
        names = {i.filename for i in infos if not i.is_dir()} - {"bundle_manifest.json"}
        if set(manifest["files"]) != names:
            raise StudyError("Bundle members differ from checksum manifest")
        # Verify all data before writing anything to the destination.
        for name, expected in manifest["files"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise StudyError(f"Bundle checksum mismatch: {name}")
            dest = target / name
            if dest.exists() and (dest.is_symlink() or file_hash(dest) != expected):
                raise StudyError(f"Destination differs: {dest}. Import into a fresh folder.")
            for parent in dest.parents:
                if parent == target.parent:
                    break
                if parent.is_symlink():
                    raise StudyError("Refusing to extract through a symlink")
        target.mkdir(parents=True,exist_ok=True)
        for name in manifest["files"]:
            dest = target / name
            if not dest.exists():
                dest.parent.mkdir(parents=True,exist_ok=True)
                with archive.open(name) as source, dest.open("wb") as stream:
                    shutil.copyfileobj(source,stream)
        write_json(target / "import_manifest.json",manifest)
    return {"files":len(manifest["files"]),"kind":manifest["kind"],"complete":manifest.get("complete",False),
            "readiness":bundle_readiness(target)}
