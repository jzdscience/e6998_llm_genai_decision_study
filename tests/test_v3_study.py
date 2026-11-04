"""Synthetic integration checks for v3 chronology and immutable evidence gates."""
from contextlib import ExitStack, contextmanager
import json
from pathlib import Path
import tempfile
import unittest
import uuid
from unittest.mock import patch
import zipfile

from decision_study.common import StudyError, fingerprint, read_json, utc_now, write_json
from decision_study.runner import _probability_comparison
from decision_study.storage import PredictionStore
from decision_study.v3 import study, cli
from decision_study.v3.identity import ROLES, interface, job_root, protocol


LABELS = ["entailment", "neutral", "contradiction"]


class StudyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="v3-synthetic-test-")
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        self.examples = []
        self.gold = {}
        for split, count in (("development", 6), ("calibration", 60), ("validation", 30), ("holdout", 18)):
            for i in range(count):
                sid = f"synthetic-{split}-{i}"
                self.examples.append({"sample_id": sid, "task": "anli", "split": split,
                    "round": f"R{i%3+1}", "group_id": f"synthetic-{split}-{i//2}",
                    "state": f"Artificial software test premise and hypothesis {i}",
                    "instructions": "Synthetic test only", "labels": LABELS,
                    "criteria": {x: x for x in LABELS}, "input_hash": fingerprint(sid)})
                self.gold[sid] = LABELS[i%3]
        self.manifest = {"evidence_type": "synthetic_software_test", "counts": {},
                         "protocol_identity": {"canonical_sha256": fingerprint(protocol())}}
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(study.data, "verify_fit", return_value=self.manifest))
        self.stack.enter_context(patch.object(study.data, "load_examples", side_effect=self.load))
        self.stack.enter_context(patch.object(study.data, "load_fit_gold", return_value={e["sample_id"]: self.gold[e["sample_id"]] for e in self.examples if e["split"] != "holdout"}))
        self.stack.enter_context(patch.object(study.runtime, "runtime_identity", side_effect=self.runtime_id))
        self.gold_reads = 0
        self.stack.enter_context(patch.object(study.data, "load_holdout_gold_for_evaluation", side_effect=self.read_holdout))

    def load(self, _output, role=None):
        if role == "fit":
            return [e for e in self.examples if e["split"] in {"calibration", "validation"}]
        return [e for e in self.examples if role is None or e["split"] == role]

    def runtime_id(self, _output, role):
        expected = protocol()["models"][role]
        effective = {"model_repo": expected["repo"], "model_revision": expected["revision"], "model": interface(role),
                     "precision": "bfloat16", "max_model_len": 8192, "prefix_cache": False, "truncate": False,
                     "backend": "vllm", "vllm_version": "0.31.0", "gpu_memory_utilization": expected["gpu_memory_utilization"]}
        if interface(role) == "qwen":
            effective.update(thinking=False, temperature=0, max_output_tokens=64, constrained_json=True)
        else:
            effective.update(think_mode="never", none_check=False, allow_none=False)
        return {"role": role, "runtime": effective}

    def read_holdout(self, _output, *, expected_freeze_hash):
        assert read_json(self.out / "holdout_access.json")["freeze_hash"] == expected_freeze_hash
        self.gold_reads += 1
        return {e["sample_id"]: self.gold[e["sample_id"]] for e in self.load(self.out, "holdout")}

    def row(self, e, role, config_hash=None):
        i = int(e["sample_id"].rsplit("-", 1)[1])
        pred = LABELS[(i + (1 if role == "startlux4b" and i%5 == 0 else 0))%3]
        probs = {x: (.65+(i%3)*.08 if x == pred else (1-(.65+(i%3)*.08))/2) for x in LABELS}
        raw = {"answers": {"decision": {"choice": pred, "probabilities": probs}}} if role == "startlux4b" else {
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps({"label": pred})}}]}
        result = {"sample_id": e["sample_id"], "task": "anli", "split": e["split"], "input_hash": e["input_hash"],
                  "model": interface(role), "status": "ok", "prediction": pred,
                  "probabilities": probs if role == "startlux4b" else None, "raw_response": raw,
                  "recorded_at": utc_now(), "attempts": [{"status": "ok", "request_id": str(uuid.uuid4())}], "latency_ms": 1}
        if config_hash:
            result["config_hash"] = config_hash
            result["model_revision"] = protocol()["models"][role]["revision"]
        return result

    def write_job(self, role, phase):
        directory = job_root(self.out, role, phase)
        config = study.config_for(self.out, role, phase)
        store = PredictionStore(directory, interface(role), config, spool_dir=self.out / "test-spool")
        for e in self.load(self.out, phase):
            store.append(self.row(e, role, fingerprint(config)))
        store.sync()
        rows = list(store.rows.values())
        dev = self.load(self.out, "development")
        smoke_rows = [self.row(e, role) for e in dev]
        write_json(directory / "smoke.json", {"config_hash": fingerprint(config), "reference": smoke_rows,
                  "current": smoke_rows, "checks": _probability_comparison(dev, smoke_rows, smoke_rows)})
        token_examples = self.load(self.out, phase) + dev
        write_json(directory / "token_preflight.json", {
            "status": "ok", "model": interface(role), "model_revision": config["runtime"]["model_revision"],
            "max_model_len": 8192, "n_examples": len(token_examples), "truncated": False, "overlong": [],
            "effective_config_hash": fingerprint(config["runtime"]),
            "rows": [{"sample_id": e["sample_id"], "input_tokens": 10,
                      "reserved_output_tokens": 64 if interface(role) == "qwen" else 1,
                      "token_ids_sha256": fingerprint(e["sample_id"])} for e in token_examples]})
        receipt = {"config_hash": fingerprint(config), "count": len(rows), "failures": 0,
                   "predictions_hash": fingerprint(sorted(rows, key=lambda r: r["sample_id"])),
                   "smoke_sha256": study._verify_smoke(self.out, directory, config),
                   "token_preflight_sha256": study._verify_tokens(self.out, directory, config)}
        if phase == "holdout":
            receipt["freeze_hash"] = read_json(self.out / "policy_freeze.json")["freeze_hash"]
        write_json(directory / "completion.json", receipt)

    def fit_all(self):
        for role in ROLES:
            self.write_job(role, "fit")
        return study.make_freeze(self.out)

    def complete_all(self):
        frozen = self.fit_all()
        for role in ROLES:
            self.write_job(role, "holdout")
        return frozen

    def test_fit_status_never_read_holdout_and_freeze_is_reproducible(self):
        first = self.fit_all()
        self.assertEqual(study.make_freeze(self.out), first)
        self.assertTrue(study.status(self.out)["freeze_present"])
        self.assertEqual(self.gold_reads, 0)

    def test_evaluation_records_access_then_is_idempotent(self):
        self.complete_all()
        result = study.evaluate(self.out)
        self.assertEqual(result["n"], 18)
        before = (self.out / "evaluation.json").read_bytes()
        self.assertEqual(result, study.evaluate(self.out))
        self.assertEqual(before, (self.out / "evaluation.json").read_bytes())
        self.assertEqual(self.gold_reads, 2)

    def test_missing_freeze_never_reads_gold(self):
        with self.assertRaises(StudyError):
            study.evaluate(self.out)
        self.assertEqual(self.gold_reads, 0)

    def test_missing_holdout_shard_never_reads_gold(self):
        self.complete_all()
        next((job_root(self.out, "qwen9b", "holdout") / "predictions/qwen").glob("shard-*.json")).unlink()
        with self.assertRaises(StudyError):
            study.evaluate(self.out)
        self.assertEqual(self.gold_reads, 0)

    def test_tampered_smoke_rejected_before_gold(self):
        self.complete_all()
        p = job_root(self.out, "startlux4b", "holdout") / "smoke.json"
        s = read_json(p); s["current"][0]["prediction"] = "invalid"; write_json(p, s)
        with self.assertRaises(StudyError): study.evaluate(self.out)
        self.assertEqual(self.gold_reads, 0)

    def test_modified_policy_cannot_reuse_holdout(self):
        self.complete_all()
        p = self.out / "policy_freeze.json"
        f = read_json(p); f["primary"]["candidate"] = "fixed_gain"; write_json(p, f)
        with self.assertRaises(StudyError): study.evaluate(self.out)
        self.assertEqual(self.gold_reads, 0)

    def test_missing_token_proof_blocks_first_gold_access(self):
        self.complete_all()
        (job_root(self.out, "qwen9b", "holdout") / "token_preflight.json").unlink()
        with self.assertRaises(FileNotFoundError): study.evaluate(self.out)
        self.assertEqual(self.gold_reads, 0)

    def test_runtime_model_change_cannot_reuse_job(self):
        self.fit_all()
        with patch.object(study.runtime, "runtime_identity", return_value={"runtime": {"model_repo": "wrong", "model_revision": "wrong"}}):
            with self.assertRaises(StudyError): study.make_freeze(self.out)

    def test_partial_export_excludes_runtime_weights(self):
        write_json(self.out / "data/example.json", {"synthetic": True})
        p = self.out / "runtime-work/cache/model.safetensors"; p.parent.mkdir(parents=True); p.write_bytes(b"test")
        with zipfile.ZipFile(study.export(self.out)) as z:
            self.assertIn("data/example.json", z.namelist())
            self.assertFalse(any("safetensors" in n for n in z.namelist()))
            self.assertNotIn("evaluation.json", z.namelist())

    def test_slurm_time_reads_current_running_job_and_reserves_five_minutes(self):
        completed = type("Result", (), {"stdout": "123|RUNNING|1-02:30:00\n"})()
        with patch.dict("os.environ", {"SLURM_JOB_ID": "123"}), patch.object(cli.subprocess, "run", return_value=completed):
            self.assertAlmostEqual(cli.slurm_hours_left(), 26.5-5/60)

    def test_slurm_rejects_bad_or_wrong_job(self):
        for output in ["123|PENDING|10:00:00", "124|RUNNING|10:00:00", "123|RUNNING|UNLIMITED"]:
            completed = type("Result", (), {"stdout": output})()
            with patch.dict("os.environ", {"SLURM_JOB_ID": "123"}), patch.object(cli.subprocess, "run", return_value=completed):
                with self.assertRaises(StudyError): cli.slurm_hours_left()


if __name__ == "__main__":
    unittest.main()
