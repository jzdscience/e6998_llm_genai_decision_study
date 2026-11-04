import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error
import zipfile
import os
import time
from contextlib import contextmanager

import numpy as np

from decision_study.adapters import ModelClient, decision_request, parse_response, qwen_request
from decision_study.artifacts import export_bundle, import_bundle, bundle_readiness
from decision_study.benchmark import cascade_call, measure
from decision_study.common import StudyError, exclusive_run, fingerprint, read_json, write_json
from decision_study.storage import PredictionStore, load_predictions
from decision_study.budget import Budget, ensure_session


EXAMPLE = {"sample_id":"x","task":"squad","split":"test","group_id":"article",
           "state":"Passage: ABC. Question: Which letters?","instructions":"Classify answerability.",
           "labels":["answerable","unanswerable"],"criteria":{"answerable":"supported","unanswerable":"unsupported"},
           "gold":"SHOULD_NEVER_ENTER_PAYLOAD"}


def response(label="answerable",p=.8):
    return {"answers":{"decision":{"choice":label,"probabilities":{"answerable":p,"unanswerable":1-p}}}}


class HTTPResponse:
    def __init__(self,value):
        self.value=value
        self.headers={}
    def __enter__(self):return self
    def __exit__(self,*_):pass
    def read(self):return json.dumps(self.value).encode()


class AdapterTests(unittest.TestCase):
    def test_gold_never_serialized_and_switches_explicit(self):
        for body in [decision_request(EXAMPLE,"kev"),decision_request(EXAMPLE,"startlux"),qwen_request(EXAMPLE)]:
            self.assertNotIn("SHOULD_NEVER_ENTER_PAYLOAD",json.dumps(body))
        self.assertFalse(decision_request(EXAMPLE,"startlux")["none_check"])
        self.assertFalse(qwen_request(EXAMPLE)["chat_template_kwargs"]["enable_thinking"])

    def test_label_mapping_and_malformed_probabilities(self):
        raw=response()
        raw["answers"]["decision"]["probabilities"]={"unanswerable":.2,"answerable":.8}
        self.assertEqual(parse_response(raw,EXAMPLE,"kev")[0],"answerable")
        for vals in ({"answerable":1.2,"unanswerable":-.2},{"answerable":.5},
                     {"answerable":float("nan"),"unanswerable":.2}, {"answerable":.7,"unanswerable":.2}):
            raw["answers"]["decision"]["probabilities"]=vals
            with self.assertRaises(StudyError):parse_response(raw,EXAMPLE,"kev")

    def test_retry_transport_once_but_not_invalid(self):
        client=ModelClient("kev","http://localhost:1",retry_delay=0)
        with patch("urllib.request.urlopen",side_effect=[urllib.error.URLError("temporary"),HTTPResponse(response())]) as mocked:
            row=client.predict(EXAMPLE)
            self.assertEqual(row["status"],"ok")
            self.assertEqual(len(row["attempts"]),2)
            self.assertEqual(mocked.call_count,2)
        with patch("urllib.request.urlopen",return_value=HTTPResponse({"bad":"shape"})) as mocked:
            self.assertEqual(client.predict(EXAMPLE)["status"],"error")
            self.assertEqual(mocked.call_count,1)

    def test_kev_serialized_16_class_mass_is_normalized_with_raw_preserved(self):
        labels = [f"label_{i}" for i in range(16)]
        example = {**EXAMPLE, "task":"clinc", "labels":labels,
                   "criteria":{label:label for label in labels}}
        # A valid unit vector [.6, .4/15, ...] serialized by official round(p,4).
        values = [.6] + [round(.4/15,4)] * 15
        raw = {"answers":{"decision":{"choice":labels[0],
                                         "probabilities":dict(zip(labels,values))}}}
        saved_raw = json.loads(json.dumps(raw))
        with patch("urllib.request.urlopen",return_value=HTTPResponse(raw)) as mocked:
            row = ModelClient("kev","http://localhost:1").predict(example)
        self.assertEqual(row["status"],"ok")
        self.assertEqual(mocked.call_count,1)
        self.assertEqual(row["raw_response"],saved_raw)
        self.assertEqual(raw,saved_raw)
        self.assertAlmostEqual(sum(row["probabilities"].values()),1.0,places=14)
        self.assertAlmostEqual(row["probabilities"][labels[0]],.6/1.0005,places=14)
        # This transport-specific allowance must not affect StartLux.
        with self.assertRaises(StudyError):
            parse_response(raw,example,"startlux")

    def test_kev_rejects_mass_outside_rounding_bound_and_unquantized_values(self):
        for values in ([.6,.3998], [.600049,.400049], [0.0,0.0]):
            raw = {"answers":{"decision":{"choice":"answerable",
                      "probabilities":dict(zip(EXAMPLE["labels"],values))}}}
            with patch("urllib.request.urlopen",return_value=HTTPResponse(raw)) as mocked:
                row = ModelClient("kev","http://localhost:1").predict(EXAMPLE)
            self.assertEqual(row["status"],"error")
            self.assertEqual(row["error"]["type"],"invalid_response")
            self.assertEqual(mocked.call_count,1)

    def test_kev_rounded_tie_preserves_server_choice_and_zero_probabilities(self):
        # Unrounded [.49996,.50004] becomes [.5,.5]; official choice stays second.
        raw = response("unanswerable",.5)
        with patch("urllib.request.urlopen",return_value=HTTPResponse(raw)):
            row = ModelClient("kev","http://localhost:1").predict(EXAMPLE)
        self.assertEqual(row["prediction"],"unanswerable")
        from decision_study.analysis import _arrays
        arrays = _arrays([EXAMPLE],{"x":"unanswerable"},{"x":row})
        self.assertTrue(arrays["valid"][0])
        self.assertTrue(arrays["correct"][0])
        self.assertEqual(parse_response(response("answerable",1.0),EXAMPLE,"kev"),
                         ("answerable",{"answerable":1.0,"unanswerable":0.0}))
        with self.assertRaises(StudyError):
            parse_response(response("unanswerable",.5001),EXAMPLE,"kev")

    def test_qwen_truncation_and_invalid_enum_fail(self):
        for reason,label in [("length","answerable"),("stop","invented")]:
            raw={"choices":[{"finish_reason":reason,"message":{"content":json.dumps({"label":label})}}]}
            with self.assertRaises(StudyError):parse_response(raw,EXAMPLE,"qwen")


class StorageTests(unittest.TestCase):
    def test_resume_keeps_terminal_failures_and_atomic_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            config={"model":"kev","precision":"bf16"}
            store=PredictionStore(root/"out","kev",config,spool_dir=root/"spool",shard_size=2)
            for i,status in enumerate(["ok","error","ok"]):
                store.append({"sample_id":str(i),"model":"kev","config_hash":fingerprint(config),"status":status})
            # Third result survives via local WAL before persistent shard sync.
            resumed=PredictionStore(root/"out","kev",config,spool_dir=root/"spool",shard_size=2)
            self.assertEqual(set(resumed.rows),{"0","1","2"})
            self.assertEqual(resumed.rows["1"]["status"],"error")
            self.assertEqual(len(load_predictions(root/"out","kev")),3)
            with self.assertRaises(StudyError):
                PredictionStore(root/"out","kev",{"model":"kev","precision":"fp16"},spool_dir=root/"spool")

    def test_incomplete_tail_recovered_but_complete_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); config={"test":True}
            store=PredictionStore(root,"kev",config,spool_dir=root/"wal")
            store.append({"sample_id":"1","model":"kev","config_hash":fingerprint(config),"status":"ok"})
            with store.journal.open("ab") as f:f.write(b'{"sample_id":')
            resumed=PredictionStore(root,"kev",config,spool_dir=root/"wal")
            self.assertIn("1",resumed.rows)
            with resumed.journal.open("ab") as f:f.write(b'garbage\n')
            with self.assertRaises(StudyError):PredictionStore(root,"kev",config,spool_dir=root/"wal")

    def test_checksum_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config={"x":1}
            store=PredictionStore(root,"kev",config,spool_dir=root/"wal")
            store.append({"sample_id":"1","model":"kev","config_hash":fingerprint(config),"status":"ok"})
            store.sync()
            path=next((root/"predictions/kev").glob("shard-*.json"))
            shard=read_json(path);shard["rows"][0]["status"]="error";write_json(path,shard)
            with self.assertRaises(StudyError):load_predictions(root,"kev")


class FakeClient:
    def __init__(self,row):self.row=row;self.calls=0
    def predict(self,example):self.calls+=1;return {**self.row,"sample_id":example["sample_id"]}


class SmokeTests(unittest.TestCase):
    def _smoke(self, root, *, parallel_matches, reference_matches=True):
        from decision_study.runner import smoke_model
        examples = [{**EXAMPLE, "sample_id":str(i), "split":"development"} for i in range(2)]
        correct = {"status":"ok", "prediction":"answerable",
                   "probabilities":{"answerable":.8,"unanswerable":.2}}
        different = {"status":"ok", "prediction":"unanswerable",
                     "probabilities":{"answerable":.2,"unanswerable":.8}}
        reference = [{**(correct if reference_matches else different), "sample_id":e["sample_id"]}
                     for e in examples]
        calls = []

        @contextmanager
        def services(*args):
            phase = "serial" if not calls else "parallel"
            calls.append(phase)
            yield {"kev":phase}

        def client(model, phase):
            return FakeClient(correct if phase == "serial" or parallel_matches else different)

        with patch("decision_study.runner.protocol_config",return_value={"fixture":True}), \
             patch("decision_study.data.load_examples",return_value=examples), \
             patch("decision_study.runtime.reference_predictions",return_value=reference), \
             patch("decision_study.runtime.managed_services",side_effect=services), \
             patch("decision_study.runner.ModelClient",side_effect=client):
            result = smoke_model(root,root,"kev")
        return result,calls

    def test_failed_parallel_trials_report_serial_only_without_fake_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            report,calls = self._smoke(Path(tmp),parallel_matches=False)
            self.assertTrue(report["passed"])
            self.assertEqual(report["selected_concurrency"],1)
            self.assertEqual(report["concurrency_status"],"serial_only")
            self.assertFalse(report["client_concurrency_verified"])
            self.assertIsNone(report["concurrency_check"])
            self.assertEqual(report["concurrent"],[])
            self.assertNotIn("batching_available",report)
            self.assertEqual([t["concurrency"] for t in report["concurrency_trials"]],[8,4])
            self.assertTrue(all(not t["checks"]["squad"]["passed"] for t in report["concurrency_trials"]))
            self.assertEqual(calls,["serial","parallel","parallel"])

    def test_matching_parallel_trial_is_reported_as_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            report,calls = self._smoke(Path(tmp),parallel_matches=True)
            self.assertTrue(report["passed"])
            self.assertEqual(report["selected_concurrency"],8)
            self.assertEqual(report["concurrency_status"],"verified")
            self.assertTrue(report["client_concurrency_verified"])
            self.assertTrue(report["concurrency_check"]["squad"]["passed"])
            self.assertEqual(len(report["concurrent"]),2)
            self.assertEqual(len(report["concurrency_trials"]),1)
            self.assertNotIn("batching_available",report)
            self.assertEqual(calls,["serial","parallel"])

    def test_serial_only_cannot_bypass_reference_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(StudyError):
                self._smoke(root,parallel_matches=True,reference_matches=False)
            report = read_json(root/"smoke/kev.json")
            self.assertFalse(report["passed"])
            self.assertFalse(report["client_concurrency_verified"])
            self.assertEqual(report["concurrency_trials"],[])
            self.assertIsNone(report["concurrency_check"])
            self.assertEqual(report["concurrent"],[])


class CascadeTests(unittest.TestCase):
    def test_failure_forces_fallback_and_fallback_failure_stays_error(self):
        error={"status":"error","prediction":None,"probabilities":None}
        local,fallback=FakeClient(error),FakeClient(error)
        cal={"temperature":{"temperature":1},"gates":{"refitted":{"accept_none":False,"threshold":.5}}}
        result=cascade_call(EXAMPLE,local,fallback,cal)
        self.assertTrue(result["deferred"])
        self.assertEqual(result["status"],"error")
        self.assertEqual(fallback.calls,1)

    def test_accepted_does_not_call_qwen(self):
        local=FakeClient({"status":"ok","prediction":"answerable","probabilities":{"answerable":.9,"unanswerable":.1}})
        fallback=FakeClient({"status":"error","prediction":None})
        cal={"temperature":{"temperature":1},"gates":{"refitted":{"accept_none":False,"threshold":.8}}}
        self.assertFalse(cascade_call(EXAMPLE,local,fallback,cal)["deferred"])
        self.assertEqual(fallback.calls,0)

    def test_gpu_seconds_count_wall_interval_once(self):
        local=FakeClient({"status":"ok","prediction":"answerable","probabilities":{"answerable":.9,"unanswerable":.1}})
        with patch("decision_study.benchmark.GPUMonitor._sample",return_value=None):
            report,rows=measure([EXAMPLE]*4,{"x":"answerable"},local,None,None,concurrency=2,min_seconds=0)
        self.assertAlmostEqual(report["allocated_gpu_seconds_per_1000"],1000*report["elapsed_seconds"]/4)
        self.assertEqual(len(rows),4)


class BundleTests(unittest.TestCase):
    def test_readiness_rejects_tampered_calibration_and_empty_benchmark(self):
        from decision_study.selftest import fixture_data
        from decision_study.analysis import fit_model
        examples,gold,rows=fixture_data()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for model in ("kev","startlux"):
                fit_model(examples,gold,rows[model],model,root)
            with patch("decision_study.data.verify_data",return_value={}),patch("decision_study.data.load_examples",return_value=examples),patch("decision_study.data.load_gold",return_value=gold),patch("decision_study.artifacts.load_predictions",side_effect=lambda r,m:rows[m]):
                self.assertTrue(bundle_readiness(root)["experiment_ready"])
                write_json(root/"benchmark/summary.json",[])
                self.assertFalse(bundle_readiness(root)["benchmark_ready"])
                path=root/"calibration/kev.json"
                cal=read_json(path);cal["tasks"]["clinc"]["temperature"]["temperature"]=1.234
                write_json(path,cal)
                self.assertFalse(bundle_readiness(root)["experiment_ready"])

    def test_incomplete_export_is_explicit_and_roundtrip_checksums(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            write_json(root/"out"/"diagnostics"/"check.json",{"error":"no GPU"})
            bundle=export_bundle(root/"out","pilot")
            imported=import_bundle(bundle,root/"restored")
            self.assertFalse(imported["complete"])
            self.assertEqual(read_json(root/"restored"/"diagnostics"/"check.json"),{"error":"no GPU"})

    def test_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);z=root/"bad.zip"
            with zipfile.ZipFile(z,"w") as f:f.writestr("../escaped.txt","bad")
            with self.assertRaises(StudyError):import_bundle(z,root/"target")
            self.assertFalse((root/"escaped.txt").exists())

    def test_runtime_installations_are_not_exported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            write_json(root/"runtime/work/envs/private/config.json",{"must_not_export":True})
            write_json(root/"runtime/lock.json",{"model":"pinned"})
            z=export_bundle(root,"pilot")
            with zipfile.ZipFile(z) as archive:
                self.assertIn("runtime/lock.json",archive.namelist())
                self.assertFalse(any("runtime/work" in n for n in archive.namelist()))


class SessionTests(unittest.TestCase):
    def test_directory_lock_prevents_concurrent_writer_and_releases(self):
        with tempfile.TemporaryDirectory() as tmp:
            with exclusive_run(Path(tmp)):
                with self.assertRaises(StudyError):
                    with exclusive_run(Path(tmp)):pass
            with exclusive_run(Path(tmp)):pass

    def test_provided_resource_deadline_persists_and_has_no_dollar_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            original=ensure_session(root,22)
            with Budget(root,"pilot",cost_mode="provided"):
                self.assertEqual(float(os.environ["DECISION_STUDY_DEADLINE_EPOCH"]),original["deadline_epoch"])
            continued=ensure_session(root,24)
            self.assertEqual(original,continued)
            ledger=read_json(root/"budget.json")
            self.assertIsNone(ledger["sessions"][0]["estimated_usd"])
            self.assertTrue(ledger["sessions"][0]["closed"])

    def test_expired_session_requires_explicit_renewal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            write_json(root/"runtime/session.json",{"deadline_epoch":time.time()-1})
            with self.assertRaises(StudyError):ensure_session(root)
            renewed=ensure_session(root,22,renew=True)
            self.assertGreater(renewed["deadline_epoch"],time.time())

    def test_existing_deadline_can_shorten_but_never_extend_without_renew(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            original=ensure_session(root,22)
            shorter=ensure_session(root,1)
            self.assertLess(shorter["deadline_epoch"],original["deadline_epoch"])
            self.assertEqual(ensure_session(root,24)["deadline_epoch"],shorter["deadline_epoch"])

    def test_nonfinite_prices_rejected_and_metered_zero_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            for value in (float("nan"),float("inf"),-1,0):
                with self.assertRaises(StudyError):Budget(Path(tmp),"pilot",value)

    def test_request_not_sent_after_deadline(self):
        with patch.dict(os.environ,{"DECISION_STUDY_DEADLINE_EPOCH":str(time.time()-1)}):
            with patch("urllib.request.urlopen") as request:
                with self.assertRaises(StudyError):ModelClient("kev","http://localhost:1").predict(EXAMPLE)
                request.assert_not_called()


if __name__ == "__main__":unittest.main()
