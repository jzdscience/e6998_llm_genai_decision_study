"""Frozen StartLux-4B -> StartLux-9B online budget and serving study (anli-budget-serving-v1).

One resumable command prepares pinned inputs, verifies the allocation and process
ownership, runs token and co-residency preflights, all 60 fixed cells, freezes the
raw evidence, then reads the already-viewed eligible-dev gold, evaluates, audits and
exports. ``runtime.py`` (service controls and supervision) is maintained separately.
"""
