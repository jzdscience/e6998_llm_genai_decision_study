"""One bounded HTTP attempt to a StartLux decision frontend.

A per-operation socket timeout does not bound a slowly dripped header or body, so
every send and every receive re-arms the socket with the time left before the
offer's absolute monotonic deadline and gives up once it has passed. The success
path is checked against the same deadline after the response is complete and
parsed; a complete but late answer keeps its raw evidence and is a deadline
failure. The request body and parser are the frozen ``adapters`` interface used by
Study 4, and each call makes at most one HTTP attempt.
"""
from __future__ import annotations

import http.client
import io
import json
import time
from urllib.parse import urlsplit
import uuid

from ..adapters import decision_request, parse_response
from ..common import StudyError, fingerprint

ROUTE = "/v1/systemone"
MAX_RESPONSE_BYTES = 1 << 20
TEXT_BYTES = 4096  # body prefix kept as raw_text for failures; the whole body when response_bytes <= TEXT_BYTES
STAGE_STATUSES = ("ok", "error", "deadline")


class Expired(Exception):
    """The absolute deadline passed while waiting for the frontend."""


class _DeadlineReader(io.RawIOBase):
    def __init__(self, sock, deadline, clock):
        super().__init__()
        self._sock, self._deadline, self._clock = sock, deadline, clock

    def readable(self):
        return True

    def readinto(self, buffer):
        left = self._deadline - self._clock()
        if left <= 0:
            raise Expired()
        self._sock.settimeout(left)
        return self._sock.recv_into(buffer)


class _DeadlineSocket:
    """Socket proxy whose sends and reads (status line, headers, body) share one deadline."""

    def __init__(self, sock, deadline, clock):
        self._sock, self._deadline, self._clock = sock, deadline, clock

    def _arm(self):
        left = self._deadline - self._clock()
        if left <= 0:
            raise Expired()
        self._sock.settimeout(left)

    def sendall(self, data, *args):
        self._arm()
        return self._sock.sendall(data, *args)

    def makefile(self, mode="rb", *args, **kwargs):
        if "b" not in mode or "w" in mode:
            raise StudyError("Unexpected socket file mode")
        return io.BufferedReader(_DeadlineReader(self._sock, self._deadline, self._clock))

    def close(self):
        # http.client closes a will-close connection right after the headers and relies on
        # socket.makefile reference counting; the caller closes the real socket after the body.
        pass

    def __getattr__(self, name):
        return getattr(self._sock, name)


def endpoint_address(endpoint):
    parts = urlsplit(endpoint)
    if parts.scheme != "http" or parts.hostname not in {"127.0.0.1", "localhost"} or parts.port is None or parts.path not in {"", "/"}:
        raise StudyError(f"Unexpected service endpoint: {endpoint!r}")
    return parts.hostname, parts.port


def call_model(role, endpoint, example, deadline, *, base, clock=time.monotonic, on_attempt=None):
    """Return a complete stage record; transport, HTTP and parse failures are data.

    ``on_attempt`` runs only once the deadline check has passed and immediately before the
    HTTP attempt starts, so a caller's attempt journal follows the actual attempt boundary.
    """
    body = decision_request(example, "startlux")
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request_id = str(uuid.uuid4())
    host, port = endpoint_address(endpoint)
    checked = clock()
    stage = {"role": role, "endpoint": endpoint, "interface": "startlux", "request_id": request_id,
             "request_sha256": fingerprint(body), "started_ms": 1000 * (checked - base), "deadline_ms": 1000 * (deadline - base),
             "http_started": False, "attempts": 0, "http_status": None, "server_request_id": None, "raw_response": None,
             "raw_text": None, "response_bytes": 0, "token_counts": None, "prediction": None, "probabilities": None,
             "late": False, "error": None}
    began = checked

    def finish(status, now=None, **fields):
        end = clock() if now is None else now
        stage.update(fields)
        stage.update(status=status, ended_ms=1000 * (end - base), latency_ms=1000 * (end - began))
        return stage

    if deadline - checked <= 0:
        return finish("deadline", checked, error={"type": "deadline_before_request"})
    if on_attempt is not None:
        on_attempt()
    began = clock()
    stage["started_ms"] = 1000 * (began - base)
    # Committed to one attempt: an expiry from here on is a started attempt that hit its deadline.
    connection = http.client.HTTPConnection(host, port, timeout=max(deadline - began, 1e-3))
    stage.update(http_started=True, attempts=1)
    data, sock = b"", None
    try:
        connection.connect()
        sock = connection.sock
        connection.sock = _DeadlineSocket(sock, deadline, clock)
        connection.request("POST", ROUTE, body=payload,
                           headers={"Content-Type": "application/json", "X-Request-ID": request_id})
        response = connection.getresponse()
        stage["http_status"] = response.status
        stage["server_request_id"] = response.getheader("x-typesafe-request-id") or response.getheader("x-request-id")
        data = response.read(MAX_RESPONSE_BYTES + 1)
    except Expired:
        return finish("deadline", error={"type": "deadline_during_http", "received_bytes": len(data)}, response_bytes=len(data))
    except TimeoutError:
        # Every socket operation is armed with exactly the time left, so a timeout is the deadline.
        return finish("deadline", error={"type": "timeout_at_deadline"}, response_bytes=len(data))
    except (OSError, http.client.HTTPException) as exc:
        return finish("error", error={"type": "transport", "message": f"{type(exc).__name__}: {exc}"}, response_bytes=len(data))
    finally:
        connection.close()
        if sock is not None:
            sock.close()
    stage["response_bytes"] = len(data)
    text = data[:TEXT_BYTES].decode("utf-8", "replace")
    if stage["http_status"] != 200:
        return finish("error", raw_text=text, error={"type": "http", "code": stage["http_status"], "message": text})
    # Every unusable HTTP 200 body is a failed call recorded as data; "phase" says where decoding stopped.
    if len(data) > MAX_RESPONSE_BYTES:
        return finish("error", error={"type": "invalid_response", "phase": "size", "message": "response exceeds byte bound"})
    try:
        decoded = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        return finish("error", raw_text=text, error={"type": "invalid_response", "phase": "utf8", "message": f"{type(exc).__name__}: {exc}"})
    try:
        raw = json.loads(decoded)
    except (ValueError, RecursionError) as exc:
        return finish("error", raw_text=text, error={"type": "invalid_response", "phase": "json", "message": f"{type(exc).__name__}: {exc}"})
    stage["raw_response"] = raw
    stage["token_counts"] = raw.get("usage") if isinstance(raw, dict) else None
    try:
        prediction, probabilities = parse_response(raw, example, "startlux")
    except StudyError as exc:
        # A JSON null body decodes to None, like an undecoded body; its text keeps the record checkable.
        return finish("error", raw_text=text if raw is None else None,
                      error={"type": "invalid_response", "phase": "answer", "message": str(exc)})
    stage.update(prediction=prediction, probabilities=probabilities)
    now = clock()
    if now > deadline:
        return finish("deadline", now, late=True, error={"type": "response_after_deadline"})
    return finish("ok", now)
