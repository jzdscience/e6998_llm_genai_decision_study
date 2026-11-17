"""Live gain of the frozen Study 4 front-feature router.

Every score is computed from the live StartLux-4B response of that very offer and the
unmodified input, through the unchanged ``modeling.front_features`` and two
``modeling.score_head`` calls. Nothing is cached by sample ID, and saved historical
scores or deferral IDs are never consulted.
"""
from __future__ import annotations

import math

import numpy as np

from ..common import StudyError
from ..learnability import modeling


def live_gain(policy, example, prediction, probabilities):
    """fallback-head minus local-head correctness probability for one live front answer."""
    sid = example["sample_id"]
    row = {"sample_id": sid, "status": "ok", "prediction": prediction, "probabilities": probabilities}
    features, failed, _ = modeling.front_features([example], {sid: row})
    if failed[0]:
        raise StudyError("A successful front answer cannot be scored as a failure")
    gain = float(modeling.score_head(policy["fallback_head"], features)[0] - modeling.score_head(policy["local_head"], features)[0])
    if not math.isfinite(gain):
        raise StudyError("Non-finite live gain")
    return gain


def batch_gains(policy, examples, fronts):
    """Vectorized recomputation for audits; repeated offers get distinct internal keys."""
    if not examples:
        return np.zeros(0)
    keyed = [{**example, "sample_id": f"offer:{i}"} for i, example in enumerate(examples)]
    rows = {f"offer:{i}": {"sample_id": f"offer:{i}", "status": "ok", "prediction": front[0], "probabilities": front[1]}
            for i, front in enumerate(fronts)}
    features, failed, _ = modeling.front_features(keyed, rows)
    if failed.any():
        raise StudyError("Audited front answers include a failure")
    return modeling.score_head(policy["fallback_head"], features) - modeling.score_head(policy["local_head"], features)
