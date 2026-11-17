"""Causal 9B-call cap: reserved(t) <= floor(numerator * offered(t) / denominator) at every instant.

The scheduler registers every actual offer under the lock before admission, including
offers the client then sheds. A worker asks for a permit only after its own front
answer; the permit is granted iff reserved < current slots, atomically with the
increment. There is no initial or future credit, no refund and no second request for
an offer. Each ledger belongs to exactly one stage or cell. The serial event journal
is the evidence: ``replay`` re-derives every counter from it.
"""
from __future__ import annotations

import math
import threading
import time

from ..common import StudyError, canonical


class InvariantViolation(StudyError):
    """An accounting or protocol invariant broke; the stage or cell must fail."""


def _slots(offered, numerator, denominator):
    return (numerator * offered) // denominator


class Ledger:
    def __init__(self, *, numerator=1, denominator=10, enforced, journal=None, clock=time.monotonic, origin=None):
        if (not isinstance(numerator, int) or not isinstance(denominator, int) or isinstance(numerator, bool)
                or isinstance(denominator, bool) or numerator < 0 or denominator < 1):
            raise StudyError("Invalid budget fraction")
        self.numerator, self.denominator, self.enforced = numerator, denominator, bool(enforced)
        self._lock = threading.Lock()
        self._journal = journal
        self._clock = clock
        self._origin = clock() if origin is None else origin
        self.offered = self.reserved = self.denied = self.sequence = 0
        self._requested = set()

    def start(self, origin):
        """Bind the stage time base before the first offer."""
        with self._lock:
            if self.sequence:
                raise InvariantViolation("Each stage needs a fresh ledger")
            self._origin = origin

    def _write(self, event):
        self.sequence += 1
        event = {"seq": self.sequence, **event, "at_seconds": self._clock() - self._origin}
        if self._journal is not None:
            self._journal.write(canonical(event) + b"\n")
            self._journal.flush()
        return event

    def offer(self, offer_id):
        """Count one actual offer before its admission decision."""
        with self._lock:
            if offer_id != self.offered:
                raise InvariantViolation("Offers must be registered once each, in scheduled order")
            self.offered += 1
            return self._write({"type": "offer", "offer_id": offer_id, "offered_received": self.offered,
                                "slots": _slots(self.offered, self.numerator, self.denominator),
                                "reserved": self.reserved, "denied": self.denied})

    def request(self, offer_id):
        """One atomic permit decision for a candidate whose offer was already counted."""
        with self._lock:
            if not self.enforced:
                raise InvariantViolation("A permit was requested in an uncapped stage")
            if not isinstance(offer_id, int) or isinstance(offer_id, bool) or not 0 <= offer_id < self.offered:
                raise InvariantViolation("Permit requested before its offer was registered")
            if offer_id in self._requested:
                raise InvariantViolation("A second permit request for one offer (retry) is not allowed")
            self._requested.add(offer_id)
            slots = _slots(self.offered, self.numerator, self.denominator)
            before = self.reserved
            granted = before < slots
            if granted:
                self.reserved += 1
            else:
                self.denied += 1
            if self.reserved > slots:
                raise InvariantViolation("Reservations exceed the causal cap")
            event = self._write({"type": "reserve" if granted else "deny", "offer_id": offer_id,
                                 "offered_at_reservation": self.offered, "slots": slots, "reserved_before": before,
                                 "reserved_after": self.reserved, "denied_after": self.denied})
            return {"budget_granted": granted, "budget_denied": not granted, "reservation_sequence": event["seq"],
                    "offered_at_reservation": self.offered, "slots_at_reservation": slots,
                    "reserved_before": before, "reserved_after": self.reserved}

    def state(self):
        with self._lock:
            return {"offered_received": self.offered, "reserved": self.reserved, "denied": self.denied,
                    "slots": _slots(self.offered, self.numerator, self.denominator), "events": self.sequence,
                    "enforced": self.enforced, "numerator": self.numerator, "denominator": self.denominator}


def replay(events, *, numerator, denominator, enforced, offer_count):
    """Independently re-derive every counter and decision from the serial journal."""
    offered = reserved = denied = 0
    decisions, last_at = {}, float("-inf")
    for index, event in enumerate(events, 1):
        kind = event.get("type")
        at = event.get("at_seconds")
        if (event.get("seq") != index or not isinstance(at, (int, float)) or isinstance(at, bool)
                or not math.isfinite(at) or at < 0 or at < last_at):
            raise InvariantViolation("Ledger sequence or clock is not serial, finite and nonnegative")
        last_at = at
        if kind == "offer":
            if event.get("offer_id") != offered:
                raise InvariantViolation("Ledger offers are not one per scheduled offer in order")
            offered += 1
            expected = {"seq": index, "type": "offer", "offer_id": offered - 1, "offered_received": offered,
                        "slots": _slots(offered, numerator, denominator), "reserved": reserved, "denied": denied, "at_seconds": at}
        elif kind in {"reserve", "deny"}:
            offer_id = event.get("offer_id")
            if not enforced or not isinstance(offer_id, int) or isinstance(offer_id, bool) or not 0 <= offer_id < offered or offer_id in decisions:
                raise InvariantViolation("Permit decision without a registered single candidate offer")
            slots = _slots(offered, numerator, denominator)
            before = reserved
            if kind == "reserve":
                if before >= slots:
                    raise InvariantViolation("Ledger granted a permit beyond the causal cap")
                reserved += 1
            else:
                if before < slots:
                    raise InvariantViolation("Ledger denied a permit while a slot was available")
                denied += 1
            decisions[offer_id] = {"granted": kind == "reserve", "seq": index, "offered_at_reservation": offered,
                                   "slots": slots, "reserved_before": before, "reserved_after": reserved}
            expected = {"seq": index, "type": kind, "offer_id": offer_id, "offered_at_reservation": offered, "slots": slots,
                        "reserved_before": before, "reserved_after": reserved, "denied_after": denied, "at_seconds": at}
        else:
            raise InvariantViolation("Unknown ledger event")
        if event != expected:
            raise InvariantViolation("Ledger event counters differ from the replay")
        if reserved > _slots(offered, numerator, denominator):
            raise InvariantViolation("Reservations exceed the causal cap")
    if offered != offer_count:
        raise InvariantViolation("Ledger did not register every scheduled offer")
    return {"offered_received": offered, "reserved": reserved, "denied": denied,
            "slots": _slots(offered, numerator, denominator), "events": len(events), "decisions": decisions}
