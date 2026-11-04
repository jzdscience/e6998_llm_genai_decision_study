"""Recorded replay and live demonstrations using the experiment's exact gate."""
from __future__ import annotations

from pathlib import Path

from .adapters import ModelClient
from .benchmark import cascade_call
from .common import StudyError, exclusive_run, fingerprint, read_json
from .storage import load_predictions


class RecordedClient:
    def __init__(self, model, rows):
        self.model, self.rows = model, {r["sample_id"]:r for r in rows}

    def predict(self, example):
        return self.rows[example["sample_id"]]


def replay(output_dir: Path, model: str = "kev", sample_id: str | None = None) -> dict:
    from .data import load_examples, load_gold
    examples, gold = load_examples(output_dir), load_gold(output_dir)
    tests = [e for e in examples if e["split"] == "test"]
    if sample_id is None:
        sample_id = min(e["sample_id"] for e in tests)
    matching = [e for e in examples if e["sample_id"] == sample_id]
    if not matching:
        raise StudyError("Unknown sample ID")
    example = matching[0]
    from .analysis import validate_calibration
    artifact = read_json(Path(output_dir)/"calibration"/f"{model}.json")
    rows = load_predictions(output_dir,model)
    validate_calibration(examples,gold,rows,artifact,model)
    cal = artifact["tasks"][example["task"]]
    result = cascade_call(example, RecordedClient(model,rows),
                          RecordedClient("qwen",load_predictions(output_dir,"qwen")),cal)
    result.pop("latency_ms")  # CPU replay is not an inference latency measurement.
    return {"mode":"RECORDED REPLAY — no model inference executed", "example":example,
            "gold":gold[sample_id],"temperature":cal["temperature"]["temperature"],
            "gate":cal["gates"]["refitted"],"result":result,
            "timing_notice":"Nested prediction timings are collection-time traces, not live cascade latency."}


def live(project_dir: Path, output_dir: Path, model: str, task: str,
         text: str = "", passage: str = "", question: str = "") -> dict:
    with exclusive_run(output_dir):
        return _live(project_dir,output_dir,model,task,text,passage,question)


def _live(project_dir: Path, output_dir: Path, model: str, task: str,
          text: str = "", passage: str = "", question: str = "") -> dict:
    from . import runtime
    from .data import BANKING_CRITERIA, BANKING_LABELS, CLINC_INSTRUCTIONS, SQUAD_CRITERIA, SQUAD_LABELS, SQUAD_INSTRUCTIONS
    from .runner import protocol_config, require_smoke
    runtime.doctor(output_dir,require_gpu=True)
    if model not in {"kev","startlux"} or task not in {"clinc","squad"}:
        raise StudyError("Choose kev/startlux and clinc/squad")
    if task == "clinc":
        if not text.strip():
            raise StudyError("Enter a banking request")
        state, labels, criteria, instruction = text, BANKING_LABELS, BANKING_CRITERIA, CLINC_INSTRUCTIONS
    else:
        if not passage.strip() or not question.strip():
            raise StudyError("SQuAD demo needs both passage and question")
        state = f"Passage:\n{passage}\n\nQuestion:\n{question}"
        labels, criteria, instruction = SQUAD_LABELS, SQUAD_CRITERIA, SQUAD_INSTRUCTIONS
    example = {"sample_id":"interactive-"+fingerprint(state)[:16],"task":task,"split":"interactive",
               "group_id":"interactive","state":state,"labels":labels,"criteria":criteria,"instructions":instruction}
    config = protocol_config(output_dir,model)
    require_smoke(output_dir,model,config)
    qwen_config = protocol_config(output_dir,"qwen")
    require_smoke(output_dir,"qwen",qwen_config)
    saved_qwen = read_json(Path(output_dir)/"predictions"/"qwen"/"config.json")
    if saved_qwen != qwen_config:
        raise StudyError("Live fallback runtime differs from collected Qwen baseline")
    calibration = read_json(Path(output_dir)/"calibration"/f"{model}.json")
    from .data import load_examples, load_gold
    from .analysis import validate_calibration
    validate_calibration(load_examples(output_dir),load_gold(output_dir),load_predictions(output_dir,model),calibration,model)
    if calibration["config_hash"] != fingerprint(config):
        raise StudyError("Demo runtime does not match the frozen gate")
    from .budget import Budget
    ledger = read_json(Path(output_dir)/"budget.json")
    last = ledger["sessions"][-1]
    with Budget(output_dir,"benchmark",last.get("hourly_usd") or 0,cost_mode=last.get("cost_mode","metered")):
        for name in (model,"qwen"):
            runtime.token_preflight_examples(project_dir,output_dir,name,[example])
        with runtime.managed_services(project_dir,output_dir,[model,"qwen"]) as endpoints:
            result = cascade_call(example,ModelClient(model,endpoints[model]),ModelClient("qwen",endpoints["qwen"]),calibration["tasks"][task])
    return {"mode":"LIVE INFERENCE","example":example,"result":result,"gold":None}
