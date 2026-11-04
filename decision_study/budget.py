"""Persistent session deadlines and monitored spending; never a cloud billing API."""
from __future__ import annotations

import math
import os
import time
from pathlib import Path
from .common import StudyError, read_json, utc_now, write_json

LIMITS = {"pilot":25.0,"prediction":75.0,"benchmark":40.0,"rerun":30.0}


def ensure_session(output_dir: Path, hours_left: float = 22, *, renew: bool = False) -> dict:
    """Keep one deadline across jobs; only an explicit new-session action resets it."""
    if not math.isfinite(hours_left) or not 0 < hours_left <= 24:
        raise StudyError("session-hours-left must be finite, greater than 0 and at most 24")
    path = Path(output_dir)/"runtime"/"session.json"
    if path.exists() and not renew:
        session = read_json(path)
        shorter_deadline = min(session["deadline_epoch"],time.time()+hours_left*3600)
        if shorter_deadline < session["deadline_epoch"]:
            session["deadline_epoch"] = shorter_deadline
            session["shortened_at"] = utc_now()
            write_json(path,session)
    else:
        session = {"started_at":utc_now(),"deadline_epoch":time.time()+hours_left*3600,
                   "hours_left_at_initialization":hours_left,
                   "notice":"User-specified remaining session allowance, not a detected provider expiry. Renew only after opening a new session."}
        write_json(path,session)
    if session["deadline_epoch"] <= time.time():
        raise StudyError("Session deadline reached. Preserve results and open a new code-server session; then use session --renew-session --session-hours-left HOURS.")
    return session


def _cost(entry):
    return entry.get("estimated_usd") or 0.0


class Budget:
    def __init__(self, output_dir: Path, stage: str, hourly_usd: float = 0, *, cost_mode: str = "metered", stage_limit: float | None = None):
        if cost_mode not in {"metered","provided"}:
            raise StudyError("cost-mode must be metered or provided")
        if not math.isfinite(hourly_usd) or hourly_usd < 0 or (cost_mode == "metered" and hourly_usd <= 0):
            raise StudyError("Metered mode requires actual HOURLY_USD > 0; company-provided mode permits 0")
        self.path = Path(output_dir)/"budget.json"
        self.session = ensure_session(output_dir)
        self.stage, self.rate, self.cost_mode = stage, hourly_usd, cost_mode
        self.limit = LIMITS[stage] if stage_limit is None else stage_limit
        self.started, self.previous_deadline = time.time(), None
        self.ledger = read_json(self.path) if self.path.exists() else {"sessions":[],"untracked_usd":0.0}
        self.entry = {"stage":stage,"started_at":utc_now(),"hourly_usd":hourly_usd if cost_mode=="metered" else None,
                      "cost_mode":cost_mode,"elapsed_seconds":0.0,"estimated_usd":0.0 if cost_mode=="metered" else None,"closed":False}
        self.ledger["sessions"].append(self.entry)
        self.ledger["notice"] = "Metered amounts cover controller-monitored time only. Provided resources have no dollar estimate, not zero economic cost. Idle/storage charges and provider expiry are not observable."
        self.check()

    def check(self):
        self.entry["elapsed_seconds"] = max(0,time.time()-self.started)
        if self.cost_mode == "metered":
            self.entry["estimated_usd"] = self.entry["elapsed_seconds"]*self.rate/3600
        write_json(self.path,self.ledger)
        if time.time() >= self.session["deadline_epoch"]:
            raise StudyError("Session time budget exhausted. Save bundle and start a new session before resuming.")
        stage_cost = sum(_cost(x) for x in self.ledger["sessions"] if x["stage"]==self.stage)
        total = sum(_cost(x) for x in self.ledger["sessions"])+self.ledger.get("untracked_usd",0)
        if self.cost_mode == "metered" and (stage_cost >= self.limit or total >= 170):
            raise StudyError("Monitored compute budget exhausted. Results preserved; stop paid VM billing in its console.")

    def estimate_remaining(self,seconds_per_input,remaining):
        seconds = seconds_per_input*remaining
        result = {"remaining_inputs":remaining,"seconds_per_input":seconds_per_input,
                  "projected_remaining_seconds":seconds,"session_seconds_left":max(0,self.session["deadline_epoch"]-time.time())}
        if self.cost_mode == "metered":
            projected = seconds*self.rate/3600
            used = sum(_cost(x) for x in self.ledger["sessions"] if x["stage"]==self.stage)
            result.update(projected_remaining_usd=projected,stage_remaining_usd=max(0,self.limit-used))
        self.ledger["last_projection"] = result
        write_json(self.path,self.ledger)
        if self.cost_mode == "metered" and result["projected_remaining_usd"] > result["stage_remaining_usd"]:
            raise StudyError("Pilot projects a stage-budget overrun; inspect budget.json before continuing")
        return result

    def close(self):
        self.entry["elapsed_seconds"] = max(0,time.time()-self.started)
        if self.cost_mode == "metered":
            self.entry["estimated_usd"] = self.entry["elapsed_seconds"]*self.rate/3600
        self.entry["closed"] = True
        write_json(self.path,self.ledger)

    def __enter__(self):
        deadline = self.session["deadline_epoch"]
        if self.cost_mode == "metered":
            used = sum(_cost(x) for x in self.ledger["sessions"] if x["stage"]==self.stage)
            total = sum(_cost(x) for x in self.ledger["sessions"])+self.ledger.get("untracked_usd",0)
            deadline = min(deadline,time.time()+max(0,min(self.limit-used,170-total))/self.rate*3600)
        self.previous_deadline = os.environ.get("DECISION_STUDY_DEADLINE_EPOCH")
        if self.previous_deadline:
            deadline = min(deadline,float(self.previous_deadline))
        os.environ["DECISION_STUDY_DEADLINE_EPOCH"] = str(deadline)
        return self

    def __exit__(self,*_):
        try:
            self.close()
        finally:
            if self.previous_deadline is None:
                os.environ.pop("DECISION_STUDY_DEADLINE_EPOCH",None)
            else:
                os.environ["DECISION_STUDY_DEADLINE_EPOCH"] = self.previous_deadline
