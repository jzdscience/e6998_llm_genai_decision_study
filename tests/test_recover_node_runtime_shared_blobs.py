"""verify_snapshot with Hugging Face's cache-wide shared blob store (huggingface_hub >= 1.32).

Real temporary caches and symlink chains. Every Xet store name differs from its repo ETag, so a check
against the wrong name fails. Each verify call is followed by a check that the cache is unchanged.
The legacy cases run against the released d1633be script and the current one; one installed-helper
case uses huggingface_hub's own writer; one CLI rehearsal goes from a shared cache to READY.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
from types import SimpleNamespace

import pytest

from test_recover_node_runtime import REPO, load, matrix_checkout, real, run_cli, tool, world  # noqa: F401 (fixtures)

RELEASED_COMMIT = "d1633be3e371c98ce491ae2a1b16bb8b935fb286"
RELEASED_SCRIPT_SHA256 = "c1d6118f8c5c757ef703f950bd536daa284528718ee18d679441e25213cb0e52"
MARKER = ".huggingface-shared-blobs"  # spelled out here, independent of the implementation's constant
WEIGHTS = bytes(range(256)) * (3 * 4096 + 7)  # just over 3 MiB: several 1 MiB hashing chunks
RESOLVE = "does not resolve to a blob of its repository ("
UNSUPPORTED, SYMLINKED, MISSING = "unsupported link", "symlinked path component", "missing or not a regular file"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def git_sha1(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def xet_of(data):
    return sha256(b"xet:" + data)  # stands in for the Xet hash: a 64-hex name that is never the ETag


def share(hub, blob, xet):
    """Move a repo blob's bytes into hub/blobs/<xet[:2]>/<xet> and link it as huggingface_hub does."""
    store = hub / "blobs"
    (store / xet[:2]).mkdir(parents=True, exist_ok=True)
    if not (store / MARKER).exists():
        (store / MARKER).write_bytes(b"1\n")
    payload = store / xet[:2] / xet
    if payload.exists():  # the same bytes were already stored for another repo
        blob.unlink()
    else:
        os.replace(blob, payload)
    os.symlink(f"../../blobs/{xet[:2]}/{xet}", blob)
    return payload


def add_repo(hub, name, files, revision="a" * 40):
    """files: relative path -> (bytes, "sha256" | "sha1" ETag, shared). Returns (snapshot, {relative: facts})."""
    repo = hub / name
    (repo / "blobs").mkdir(parents=True, exist_ok=True)
    snapshot, facts = repo / "snapshots" / revision, {}
    for relative, (data, kind, shared) in files.items():
        etag = sha256(data) if kind == "sha256" else git_sha1(data)
        blob = repo / "blobs" / etag
        payload = blob
        if not os.path.lexists(blob):
            blob.write_bytes(data)
            if shared:
                payload = share(hub, blob, xet_of(data))
        link = snapshot / relative
        link.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(os.path.relpath(blob, link.parent), link)
        facts[relative] = SimpleNamespace(etag=etag, xet=xet_of(data), blob=blob, payload=payload, data=data)
    return snapshot, facts


def tree_state(root):
    """Everything verify could change, read without following links (atime excluded)."""
    state = {}
    for directory, folders, files in os.walk(root):
        for name in folders + files:
            path = Path(directory, name)
            info = os.lstat(path)
            if stat.S_ISLNK(info.st_mode):
                content = os.readlink(path)
            elif stat.S_ISREG(info.st_mode):
                content = sha256(path.read_bytes())
            else:
                content = None
            state[path.relative_to(root).as_posix()] = (stat.S_IFMT(info.st_mode), stat.S_IMODE(info.st_mode),
                                                        info.st_size, info.st_mtime_ns, info.st_ctime_ns, content)
    return state


def verify(module, snapshot, base, label="probe"):
    """verify_snapshot plus the guarantee that it wrote nothing anywhere below ``base``."""
    before, problems = tree_state(base), []
    checked = module.verify_snapshot(snapshot, label, problems)
    assert tree_state(base) == before, "verify_snapshot changed the cache"
    return checked, problems


def relink(path, target):
    os.unlink(path)
    os.symlink(target, path)


# --------------------------------------------------------------------------- accepted layouts

def test_legacy_shared_and_mixed_layouts_verify_against_the_etag(tmp_path):
    base = real(tmp_path)
    hub = base / "hub"
    index = json.dumps({"weight_map": {"x": "model-00001-of-00002.safetensors",
                                       "y": "model-00002-of-00002.safetensors"}}).encode()
    legacy, _ = add_repo(hub, "models--org--legacy", {"model.safetensors": (b"legacy weights", "sha256", False),
                                                      "config.json": (b"{}", "sha1", False)})
    mixed, a = add_repo(hub, "models--org--mixed", {
        "model-00001-of-00002.safetensors": (WEIGHTS, "sha256", True),
        "model-00002-of-00002.safetensors": (b"second shard", "sha256", False),
        "model.safetensors.index.json": (index, "sha1", False),
        "tokenizer.json": (b'{"vocab": 1}', "sha1", True),  # a 40-hex ETag, hashed as a Git blob over the payload
        "sub/head.pt": (b"head", "sha256", True),  # nested entry: ../../../blobs/<etag>
    })
    shared, b = add_repo(hub, "models--org--shared", {"model.safetensors": (WEIGHTS, "sha256", True)}, revision="b" * 40)
    payload = a["model-00001-of-00002.safetensors"].payload
    assert payload == b["model.safetensors"].payload == hub / "blobs" / xet_of(WEIGHTS)[:2] / xet_of(WEIGHTS)
    assert payload.name != sha256(WEIGHTS) and os.readlink(b["model.safetensors"].blob).startswith("../../blobs/")
    assert [verify(tool, s, base) for s in (legacy, mixed, shared)] == [(2, []), (5, []), (1, [])]


def test_layout_written_by_huggingface_hub_is_accepted_and_agrees_with_its_reader(tmp_path):
    hf = pytest.importorskip("huggingface_hub.utils._shared_blobs")
    base = real(tmp_path)
    hub, etag, xet = base / "hub", sha256(WEIGHTS), xet_of(WEIGHTS)
    snapshots, blobs = [], []
    for name in ("models--org--first", "models--org--second"):
        (hub / name / "blobs").mkdir(parents=True)
        snapshot = hub / name / "snapshots" / ("c" * 40)
        snapshot.mkdir(parents=True)
        os.symlink(f"../../blobs/{etag}", snapshot / "model.safetensors")
        snapshots.append(snapshot)
        blobs.append(hub / name / "blobs" / etag)
    blobs[0].write_bytes(WEIGHTS)
    assert hf.publish_blob_to_shared_store(blob_path=str(blobs[0]), xet_hash=xet, cache_dir=hub,
                                           expected_size=len(WEIGHTS))
    assert hf.try_link_from_shared_store(blob_path=str(blobs[1]), xet_hash=xet, cache_dir=hub,
                                         expected_size=len(WEIGHTS))
    payload = hub / "blobs" / xet[:2] / xet
    assert os.readlink(blobs[0]) == os.readlink(blobs[1]) == f"../../blobs/{xet[:2]}/{xet}"  # the fixture form above
    assert (hub / "blobs" / MARKER).read_bytes() == b"1\n" and hf.is_shared_blobs_dir(hub / "blobs")
    assert [verify(tool, s, base) for s in snapshots] == [(1, []), (1, [])]  # with HF's .refs and .lock files present
    for blob in blobs:
        assert tool.hub_blob_payload(blob) == hf.shared_blob_target(blob, hub) == payload


# --------------------------------------------------------------------------- the ETag decides, never the Xet name

def shared_case(tmp_path, repo_name="models--org--model"):
    base = real(tmp_path)
    hub = base / "hub"
    snapshot, facts = add_repo(hub, repo_name, {"model.safetensors": (WEIGHTS, "sha256", True),
                                                "config.json": (b"{}", "sha1", False)})
    m = facts["model.safetensors"]
    return SimpleNamespace(base=base, hub=hub, repo=hub / repo_name, snapshot=snapshot, entry=snapshot / "model.safetensors",
                           blob=m.blob, payload=m.payload, etag=m.etag, xet=m.xet, x2=m.xet[:2])


def test_corrupt_shared_payload_fails_its_etag(tmp_path):
    c = shared_case(tmp_path)
    c.payload.write_bytes(b"X" + WEIGHTS[1:])
    assert verify(tool, c.snapshot, c.base) == (2, ["probe: blob content does not match its name: model.safetensors"])


def test_payload_named_by_its_own_sha256_still_fails_the_etag(tmp_path):
    """A store file whose name is the SHA-256 of its (wrong) bytes: only an ETag comparison catches it."""
    c = shared_case(tmp_path)
    wrong = b"other weights"
    name = sha256(wrong)
    (c.hub / "blobs" / name[:2]).mkdir(exist_ok=True)
    (c.hub / "blobs" / name[:2] / name).write_bytes(wrong)
    c.payload.unlink()
    relink(c.blob, f"../../blobs/{name[:2]}/{name}")
    assert verify(tool, c.snapshot, c.base) == (2, ["probe: blob content does not match its name: model.safetensors"])


# --------------------------------------------------------------------------- everything else is refused

def move_payload(c, relative):
    target = c.hub / "blobs" / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(c.payload, target)
    relink(c.blob, f"../../blobs/{relative}")


def displace(path, outside):
    """Swap ``path`` for a link to the same content outside the cache: it resolves, but escapes."""
    outside.parent.mkdir(parents=True, exist_ok=True)
    os.replace(path, outside)
    os.symlink(outside, path)


def other_cache(c):
    store = c.base / "other-cache" / "blobs"
    (store / c.x2).mkdir(parents=True)
    (store / MARKER).write_bytes(b"1\n")
    (store / c.x2 / c.xet).write_bytes(WEIGHTS)
    relink(c.blob, f"../../../other-cache/blobs/{c.x2}/{c.xet}")


def payload_fifo(c):  # refused from lstat alone; a FIFO is never opened
    c.payload.unlink()
    os.mkfifo(c.payload)


def cycle(c):
    c.payload.unlink()
    os.symlink(f"../../{c.repo.name}/blobs/{c.etag}", c.payload)


def extra_hop(c):
    final = c.payload.with_name(c.x2 + "f" * 62)
    os.replace(c.payload, final)
    os.symlink(final.name, c.payload)


def rename_repo(c):
    os.rename(c.repo, c.hub / "org--model")
    return c.hub / "org--model" / "snapshots" / c.snapshot.name


REFUSED = {
    "absolute-target-to-the-same-payload": (lambda c: relink(c.blob, str(c.payload)), UNSUPPORTED),
    "other-cache-root": (other_cache, UNSUPPORTED),
    "prefix-not-hash-start": (lambda c: move_payload(c, f"{'00' if c.x2 != '00' else '11'}/{c.xet}"), UNSUPPORTED),
    "uppercase-hash": (lambda c: move_payload(c, f"{c.x2.upper()}/{c.xet.upper()}"), UNSUPPORTED),
    "63-hex-hash": (lambda c: move_payload(c, f"{c.x2}/{c.xet[:63]}"), UNSUPPORTED),
    "extra-path-component": (lambda c: move_payload(c, f"{c.x2}/{c.x2}/{c.xet}"), UNSUPPORTED),
    "dot-dot-inside-target": (lambda c: relink(c.blob, f"../../blobs/{c.x2}/../{c.x2}/{c.xet}"), UNSUPPORTED),
    "not-a-hub-repository-folder": (rename_repo, UNSUPPORTED),
    "dangling-payload": (lambda c: c.payload.unlink(), MISSING),
    "payload-is-a-fifo": (payload_fifo, MISSING),
    "cycle-back-to-repo-blob": (cycle, SYMLINKED),
    "extra-symlink-hop": (extra_hop, SYMLINKED),
    "store-directory-escape": (lambda c: displace(c.hub / "blobs", c.base / "outside" / "blobs"), SYMLINKED),
    "prefix-directory-escape": (lambda c: displace(c.hub / "blobs" / c.x2, c.base / "outside" / c.x2), SYMLINKED),
    "marker-missing": (lambda c: (c.hub / "blobs" / MARKER).unlink(), MISSING),
    "marker-other-version": (lambda c: (c.hub / "blobs" / MARKER).write_bytes(b"2\n"), "not a version-1"),
    "marker-escape": (lambda c: displace(c.hub / "blobs" / MARKER, c.base / "outside" / MARKER), SYMLINKED),
    "snapshot-link-skips-repo-blob": (lambda c: relink(c.entry, f"../../../blobs/{c.x2}/{c.xet}"),
                                      "leaves the repository's blobs directory"),
}


@pytest.mark.parametrize("case", sorted(REFUSED))
def test_unsupported_links_and_escapes_are_refused(tmp_path, case):
    c = shared_case(tmp_path)
    mutate, reason = REFUSED[case]
    moved = mutate(c)  # only rename_repo returns a (new) snapshot path
    checked, problems = verify(tool, moved if isinstance(moved, Path) else c.snapshot, c.base)
    assert checked == 1, problems  # config.json, a regular repo blob, still verifies
    assert len(problems) == 1 and problems[0].startswith(f"probe: model.safetensors {RESOLVE}"), problems
    assert reason in problems[0], problems


# --------------------------------------------------------------------------- legacy behaviour against the release

@pytest.fixture(scope="module")
def released(tmp_path_factory):
    source = subprocess.run(["git", "-C", str(REPO), "show", f"{RELEASED_COMMIT}:scripts/recover_node_runtime.py"],
                            check=True, capture_output=True).stdout
    assert sha256(source) == RELEASED_SCRIPT_SHA256
    path = tmp_path_factory.mktemp("released") / "recover_node_runtime_released.py"
    path.write_bytes(source)
    return load(path)


def legacy_case(tmp_path):
    base = real(tmp_path)
    hub = base / "hub"
    snapshot, facts = add_repo(hub, "models--org--model", {"model.safetensors": (WEIGHTS, "sha256", False),
                                                           "config.json": (b"{}", "sha1", False),
                                                           "sub/extra.safetensors": (b"extra", "sha256", False)})
    m = facts["model.safetensors"]
    return SimpleNamespace(base=base, hub=hub, repo=hub / "models--org--model", snapshot=snapshot,
                           entry=snapshot / "model.safetensors", blob=m.blob, etag=m.etag)


def sibling_hop(c):
    final = c.blob.with_name("f" * 64)
    os.replace(c.blob, final)
    os.symlink(final.name, c.blob)


def other_repo(c):
    (c.hub / "models--org--other/blobs").mkdir(parents=True)
    os.replace(c.blob, c.hub / "models--org--other/blobs" / c.etag)
    relink(c.entry, f"../../../models--org--other/blobs/{c.etag}")


def unexpected_name(c):
    os.replace(c.blob, c.blob.with_name("README"))
    relink(c.entry, "../../blobs/README")


def entry_not_a_link(c):
    c.entry.unlink()
    c.entry.write_bytes(WEIGHTS)


LEGACY_SAME = {
    "valid": lambda c: None,
    "corrupt-blob": lambda c: c.blob.write_bytes(b"X" + WEIGHTS[1:]),
    "blob-links-to-a-sibling": sibling_hop,
    "blob-of-another-repository": other_repo,
    "unexpected-blob-name": unexpected_name,
    "entry-not-a-link": entry_not_a_link,
    "dangling-entry": lambda c: c.blob.unlink(),
}


def outcome(module, c):
    checked, problems = verify(module, c.snapshot, c.base)
    return checked, [p.split(" (", 1)[0] for p in problems]  # the current script appends a reason in parentheses


@pytest.mark.parametrize("case", sorted(LEGACY_SAME))
def test_legacy_layout_outcomes_equal_the_released_script(tmp_path, released, case):
    c = legacy_case(tmp_path)
    LEGACY_SAME[case](c)
    assert outcome(tool, c) == outcome(released, c)


def test_two_intended_differences_from_the_released_script(tmp_path, released):
    shared = shared_case(tmp_path / "shared")  # the node layout: released refuses, current accepts
    assert outcome(released, shared) == (1, [f"probe: model.safetensors {RESOLVE[:-2]}"])
    assert outcome(tool, shared) == (2, [])
    escaped = legacy_case(tmp_path / "escaped")  # a repo blobs directory linked out of the cache: now refused
    displace(escaped.repo / "blobs", escaped.base / "outside" / "blobs")
    assert outcome(released, escaped) == (3, [])
    checked, problems = verify(tool, escaped.snapshot, escaped.base)
    assert checked == 0 and len(problems) == 3 and all(RESOLVE in p and SYMLINKED in p for p in problems), problems


# --------------------------------------------------------------------------- CLI rehearsal: shared caches reach READY

HUBS = ("results/solo-v1/runtime/work/cache/huggingface/hub", "results/solo-v3/runtime-work/cache/huggingface/hub",
        "results/aux-size-control-v1-r2/runtime-work/cache/huggingface/hub")


def share_cache(hub):
    """Every 64-hex repo blob of a rehearsal cache moved into its shared store; returns {etag: payload}."""
    return {blob.name: share(hub, blob, xet_of(blob.read_bytes())) for blob in sorted(hub.glob("models--*/blobs/*"))
            if re.fullmatch(r"[0-9a-f]{64}", blob.name) and not blob.is_symlink()}


def test_rehearsal_shared_caches_reach_ready_and_a_corrupt_payload_blocks(world):
    w = world
    code, report = run_cli(w, "execute")
    assert (code, report["status"]) == (0, "MATRIX_RUNTIME_READY"), report.get("problems") or report
    counts = {label: snapshot["blobs_checked"] for label, snapshot in report["snapshots"].items()}
    payloads = {relative: share_cache(w.root / relative) for relative in HUBS}
    assert all(payloads.values()) and all((w.root / relative / "blobs" / MARKER).is_file() for relative in HUBS)
    caches = lambda: {relative: tree_state(w.root / relative) for relative in HUBS}
    before = caches()
    code, report = run_cli(w, "verify")
    assert (code, report["status"]) == (0, "MATRIX_RUNTIME_READY"), report.get("problems")
    assert {label: snapshot["blobs_checked"] for label, snapshot in report["snapshots"].items()} == counts
    assert caches() == before, "verify wrote into a Hugging Face cache"
    payloads[HUBS[1]][sha256(b"w1")].write_bytes(b"wX")  # Qwen3.5-9B model-1.safetensors, same size
    code, report = run_cli(w, "verify")
    assert (code, report["status"]) == (1, "MATRIX_RUNTIME_NOT_READY")
    assert report["problems"] == ["qwen9b: blob content does not match its name: model-1.safetensors"]
