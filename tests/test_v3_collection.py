"""Real local HTTP -> BoundClient -> predict_examples -> PredictionStore tests.

Only synthetic examples and responses are used. Runtime/model execution, dataset
I/O and source-identity snapshots are mocked; the collector/HTTP/parser/store,
policy fitting, thresholding and statistics execute normally. No GPU is used.
"""
from collections import Counter
from contextlib import ExitStack, contextmanager
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import tempfile
import threading
import unittest
from unittest.mock import patch

from decision_study.common import StudyError, fingerprint, read_json, write_json
from decision_study.storage import load_predictions
from decision_study.v3 import study, cli
from decision_study.v3.identity import ROLES, interface, job_root, protocol

LABELS = ["entailment", "neutral", "contradiction"]


class FakeBudget:
    def __init__(self):
        self.calls, self.fail_after = 0, None
        self.estimates = []

    def check(self):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise StudyError("Synthetic budget interruption")

    def estimate_remaining(self, seconds, count):
        self.estimates.append((seconds, count))


class CollectionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="v3-http-collection-")
        self.addCleanup(temporary.cleanup)
        self.out = Path(temporary.name)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.examples, self.gold = [], {}
        for split, count in (("development",3),("calibration",30),("validation",20),("holdout",9)):
            for i in range(count):
                sid = f"synthetic-{split}-{i}"
                example = {"sample_id":sid,"task":"anli","split":split,"round":f"R{i%3+1}",
                           "group_id":f"synthetic-{split}-premise-{i//2}","state":f"Artificial test input {sid}",
                           "instructions":"Software fixture only", "labels":list(LABELS),
                           "criteria":{label:label for label in LABELS}}
                example["input_hash"] = fingerprint(example)
                self.examples.append(example)
                self.gold[sid] = LABELS[i%3]
        self.manifest = {"counts":{},"evidence_type":"synthetic_software_test",
                         "protocol_identity":{"canonical_sha256":fingerprint(protocol())}}
        self.requests = Counter()
        self.transient, self.invalid = set(), set()
        self.budgets, self.service_starts = [], []
        self.stop_after, self.gold_reads, self.current_role = None, 0, None
        # Freeze this test's identity snapshot so other agents editing unrelated
        # v3 modules during development cannot fabricate a resume mismatch.
        sources = study.source_identity()
        self.stack.enter_context(patch.object(study,"source_identity",return_value=sources))
        self.stack.enter_context(patch.object(study.data,"verify_fit",return_value=self.manifest))
        self.stack.enter_context(patch.object(study.data,"load_examples",side_effect=self.load))
        self.stack.enter_context(patch.object(study.data,"load_fit_gold",return_value={e["sample_id"]:self.gold[e["sample_id"]] for e in self.examples if e["split"]!="holdout"}))
        self.stack.enter_context(patch.object(study.data,"load_holdout_gold_for_evaluation",side_effect=self.read_gold))
        self.stack.enter_context(patch.object(study.runtime,"runtime_identity",side_effect=self.runtime_id))
        self.stack.enter_context(patch.object(study.runtime,"runtime_context",side_effect=self.runtime_context))
        self.stack.enter_context(patch.object(study.runtime,"ensure_download",return_value={"synthetic":True}))
        self.token_mock = self.stack.enter_context(patch.object(study.runtime,"token_preflight",side_effect=self.tokens))
        self.stack.enter_context(patch.object(study.runtime,"managed_services",side_effect=self.services))
        owner = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                state = body.get("state") or body["messages"][-1]["content"]
                match = re.search(r"synthetic-(development|calibration|validation|holdout)-(\d+)",state)
                if not match:
                    self.send_error(400,"Unknown synthetic input")
                    return
                sid, i = match.group(0), int(match.group(2))
                role = owner.current_role
                model = interface(role)
                key = (role,sid)
                owner.requests[key] += 1
                if key in owner.transient and owner.requests[key] == 1:
                    self.send_error(503,"Synthetic retryable failure")
                    return
                label = LABELS[(i + int(role=="startlux4b" and i%5==0))%3]
                confidence = .65 + .08*(i%3)
                if model == "startlux":
                    assert self.path == "/v1/systemone"
                    assert body["allow_none"] is False
                    response = {"answers":{"decision":{"choice":label,
                        "probabilities":{x:confidence if x==label else (1-confidence)/2 for x in LABELS}}},
                        "model":"startlux","usage":{"input_tokens":10,"output_tokens":0}}
                else:
                    assert self.path == "/v1/chat/completions"
                    assert body["structured_outputs"]["json"]["properties"]["label"]["enum"] == LABELS
                    response = {"choices":[{"finish_reason":"stop","message":{"content":json.dumps({"label":label})}}],
                                "model":"qwen","id":self.headers["X-Request-ID"],"usage":{"prompt_tokens":10,"completion_tokens":1}}
                if key in owner.invalid:
                    response = {"invalid":"synthetic terminal parse failure"}
                data = json.dumps(response).encode()
                self.send_response(200)
                self.send_header("Content-Type","application/json")
                self.send_header("Content-Length",str(len(data)))
                self.send_header("X-Request-ID",self.headers["X-Request-ID"])
                self.end_headers()
                self.wfile.write(data)
        self.server = ThreadingHTTPServer(("127.0.0.1",0),Handler)
        self.thread = threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()
        def stop():
            self.server.shutdown()
            self.server.server_close()
            self.thread.join(timeout=5)
        self.addCleanup(stop)
        self.endpoint = f"http://127.0.0.1:{self.server.server_port}"

    def load(self,_output,role=None):
        return [e for e in self.examples if role is None or (e["split"] in {"calibration","validation"} if role=="fit" else e["split"]==role)]

    def runtime_id(self,_output,role):
        expected=protocol()["models"][role]
        value={"model":interface(role),"model_repo":expected["repo"],"model_revision":expected["revision"],
               "precision":"bfloat16","max_model_len":8192,"prefix_cache":False,"chunked_prefill":False,
               "truncate":False,"gpu_memory_utilization":expected["gpu_memory_utilization"],
               "thinking":False,"temperature":0,"max_output_tokens":64,"think_mode":"never",
               "backend":"vllm","vllm_version":"0.31.0","constrained_json":True,
               "allow_none":False,"none_check":False,"bidirectional":True,"speculative_decoding":False}
        return {"role":role,"runtime":value}

    def tokens(self,_output,role,examples,*,control=None):
        identity=self.runtime_id(_output,role)["runtime"]
        return {"status":"ok","model":interface(role),"model_revision":identity["model_revision"],
                "max_model_len":8192,"n_examples":len(examples),"rows":[
                    {"sample_id":e["sample_id"],"input_tokens":10,
                     "reserved_output_tokens":64 if interface(role)=="qwen" else 1,
                     "token_ids_sha256":fingerprint([1]*10)} for e in examples],
                "overlong":[],"renderer":"official","truncated":False,
                "effective_config_hash":fingerprint(identity)}

    @contextmanager
    def runtime_context(self,_v1,_output,role):
        budget=FakeBudget()
        budget.fail_after=self.stop_after
        self.budgets.append(budget)
        yield self.out/"fake-control"/role,budget

    @contextmanager
    def services(self,_output,role,*,control=None):
        self.current_role=role
        self.service_starts.append(role)
        yield self.endpoint

    def read_gold(self,_output,*,expected_freeze_hash):
        # Synthetic only: prove collector and all three completion gates ran first.
        self.assertEqual(read_json(self.out/"holdout_access.json")["freeze_hash"],expected_freeze_hash)
        for role in ROLES:
            self.assertTrue((job_root(self.out,role,"holdout")/"completion.json").exists())
        self.gold_reads+=1
        return {e["sample_id"]:self.gold[e["sample_id"]] for e in self.load(self.out,"holdout")}

    def run_job(self,role="startlux4b",phase="fit"):
        return study.run_job(self.out/"unused-v1",self.out,role,phase)

    def test_real_collector_completes_all_interfaces_and_skips_completed_jobs(self):
        for role in ROLES:
            done=self.run_job(role)
            self.assertEqual((done["count"],done["sampling"]["new"],done["failures"]),(50,50,0))
            rows,_=study.verified_job(self.out,role,"fit")
            self.assertEqual(len(rows),50)
            self.assertTrue(all(r["attempts"][0]["request_id"]==r["attempts"][0]["server_request_id"] for r in rows))
            self.assertTrue(all(r["model_revision"]==protocol()["models"][role]["revision"] for r in rows))
            calls=copy.copy(self.requests)
            starts=len(self.service_starts)
            self.assertEqual(self.run_job(role),done)
            self.assertEqual(self.requests,calls)
            self.assertEqual(len(self.service_starts),starts)
        self.assertTrue(all(budget.estimates for budget in self.budgets))
        self.assertEqual(self.gold_reads,0)

    def test_interruption_syncs_and_resume_requests_only_missing_examples(self):
        self.stop_after=6+7  # Two development smoke passes, then seven formal rows.
        with self.assertRaisesRegex(StudyError,"Synthetic budget"):
            self.run_job()
        directory=job_root(self.out,"startlux4b","fit")
        self.assertFalse((directory/"completion.json").exists())
        self.assertEqual(len(load_predictions(directory,"startlux")),7)
        shard=next((directory/"predictions/startlux").glob("shard-*.json"))
        original=shard.read_bytes()
        self.stop_after=None
        done=self.run_job()
        self.assertEqual(done["sampling"]["new"],43)
        self.assertEqual(shard.read_bytes(),original)
        for e in self.load(self.out,"fit"):
            self.assertEqual(self.requests[("startlux4b",e["sample_id"])],1)
        self.assertEqual(self.gold_reads,0)

    def test_retry_is_persisted_with_distinct_request_ids(self):
        self.transient.add(("qwen9b","synthetic-calibration-0"))
        done=self.run_job("qwen9b")
        rows,_=study.verified_job(self.out,"qwen9b","fit")
        row=next(r for r in rows if r["sample_id"]=="synthetic-calibration-0")
        self.assertEqual([a["status"] for a in row["attempts"]],["error","ok"])
        self.assertNotEqual(row["attempts"][0]["request_id"],row["attempts"][1]["request_id"])
        self.assertEqual(done["failures"],0)

    def test_preflight_failure_prevents_http_and_formal_store(self):
        self.token_mock.side_effect=None
        self.token_mock.return_value={"status":"blocked"}
        with self.assertRaisesRegex(StudyError,"token"):
            self.run_job()
        self.assertFalse(self.requests)
        self.assertFalse(self.service_starts)
        self.assertFalse((job_root(self.out,"startlux4b","fit")/"predictions").exists())

    def test_smoke_failure_prevents_formal_collection(self):
        self.invalid.add(("startlux4b","synthetic-development-0"))
        with self.assertRaisesRegex(StudyError,"Smoke"):
            self.run_job()
        self.assertTrue(all("development" in sid for _,sid in self.requests))
        self.assertEqual(load_predictions(job_root(self.out,"startlux4b","fit"),"startlux"),[])

    def test_collected_pipeline_freezes_before_holdout_and_access(self):
        with self.assertRaisesRegex(StudyError,"freeze"):
            self.run_job("qwen9b","holdout")
        self.assertFalse(self.requests)
        for role in ROLES:
            self.run_job(role)
        # Existing fit jobs alone must not silently create holdout authorization.
        before=copy.copy(self.requests)
        with self.assertRaisesRegex(StudyError,"freeze"):
            self.run_job("qwen9b","holdout")
        self.assertFalse((self.out/"policy_freeze.json").exists())
        self.assertEqual(self.requests,before)
        frozen=study.make_freeze(self.out)
        self.assertEqual(frozen["primary"],{"fallback":"qwen9b","candidate":"positive_gain","reference":"confidence"})
        for role in ROLES:
            self.run_job(role,"holdout")
            self.assertEqual(self.gold_reads,0)
        result=study.evaluate(self.out)
        self.assertEqual(result["n"],9)
        self.assertEqual(result["primary_difference"]["replicates"],2000)
        self.assertEqual(self.gold_reads,1)
        self.assertEqual(study.evaluate(self.out),result)

    def test_wrong_token_input_id_blocks_before_any_http(self):
        def wrong(*args,**kwargs):
            report=self.tokens(*args,**kwargs)
            report["rows"][0]["sample_id"]="not-a-supplied-input"
            return report
        self.token_mock.side_effect=wrong
        with self.assertRaises(StudyError):
            self.run_job()
        self.assertFalse(self.requests)
        self.assertFalse(self.service_starts)

    def test_missing_bound_token_report_refuses_completed_reuse(self):
        self.run_job()
        path=job_root(self.out,"startlux4b","fit")/"token_preflight.json"
        self.assertTrue(path.exists())
        path.unlink()
        before=copy.copy(self.requests)
        with self.assertRaises((StudyError,FileNotFoundError)):
            self.run_job()
        self.assertEqual(self.requests,before)


class PipelineOrderTests(unittest.TestCase):
    def test_cli_pipeline_orders_every_phase_and_exports(self):
        events=[]
        with tempfile.TemporaryDirectory(prefix="v3-cli-order-") as tmp, ExitStack() as stack:
            stack.enter_context(patch.object(cli,"verify_release",return_value={"status":"verified"}))
            stack.enter_context(patch.object(study,"run_job",side_effect=lambda v1,out,role,phase:events.append((phase,role))))
            stack.enter_context(patch.object(study,"make_freeze",side_effect=lambda out:events.append(("freeze",))))
            stack.enter_context(patch.object(study,"evaluate",side_effect=lambda out:events.append(("evaluate",))))
            stack.enter_context(patch.object(study,"export",side_effect=lambda out:events.append(("export",))))
            self.assertEqual(cli.main(["pipeline","--output-dir",tmp]),0)
        expected=[item for role in ROLES for item in (("fit",role),("export",))]
        expected += [("freeze",)]
        expected += [item for role in ROLES for item in (("holdout",role),("export",))]
        expected += [("evaluate",),("export",)]
        self.assertEqual(events,expected)

    def test_cli_pipeline_stops_after_failed_fit_and_exports_diagnostic(self):
        events=[]
        def collect(v1,out,role,phase):
            events.append((phase,role))
            if role=="qwen4b":
                raise StudyError("Synthetic collection failure")
        with tempfile.TemporaryDirectory(prefix="v3-cli-stop-") as tmp, ExitStack() as stack:
            stack.enter_context(patch.object(cli,"verify_release",return_value={"status":"verified"}))
            stack.enter_context(patch.object(study,"run_job",side_effect=collect))
            freeze=stack.enter_context(patch.object(study,"make_freeze"))
            evaluate=stack.enter_context(patch.object(study,"evaluate"))
            exporter=stack.enter_context(patch.object(study,"export",return_value=Path(tmp)/"fake-diagnostic.zip"))
            self.assertEqual(cli.main(["pipeline","--output-dir",tmp]),2)
            self.assertEqual(read_json(Path(tmp)/"diagnostics/last_error.json")["error"],"Synthetic collection failure")
            self.assertEqual(exporter.call_count,2)  # after first fit, then diagnostic
            freeze.assert_not_called()
            evaluate.assert_not_called()
        self.assertEqual(events,[("fit","startlux4b"),("fit","qwen4b")])


if __name__ == "__main__":
    unittest.main()
