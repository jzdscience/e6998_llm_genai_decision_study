"""Independent acceptance cases that remove incidental blockers from two mutation checks."""
import json

from test_recover_node_runtime import (
    calls, matrix_checkout, run_cli, state_of, tool, tree_listing, world,
)


def test_tampered_metadata_annotation_blocks_resume_before_any_installer(world):
    """A benign-looking extra field keeps all executable paths valid; only the pin gate stops it."""
    w = world
    (w.fake / "control/fail-install-once-vllm-0.31.0").touch()
    run_cli(w, "execute")
    assert state_of(w)["phases"]["metadata"]["status"] == "complete"
    installations = w.root / tool.V1_RUNTIME / "installations.json"
    record = json.loads(installations.read_text())
    record["unexpected_annotation"] = "accepted metadata bytes must stay unchanged"
    installations.write_text(json.dumps(record))
    before = len(calls(w))
    code, report = run_cli(w, "execute")
    invoked = [c for c in calls(w)[before:] if c["args"] != ["--version"]]
    assert invoked == [], "metadata drift reached an installer or metadata-derived program"
    assert (code, report["status"]) == (2, "FAILED")
    assert "fail their pins" in report["error"]


def test_verify_tmpdir_inside_clean_rehearsal_root_never_creates_a_directory(world):
    """The matrix checkout is clean, so it cannot hide an omitted temporary-root guard."""
    w = world
    code, report = run_cli(w, "plan")
    assert (code, report["status"]) == (0, "READY_TO_EXECUTE")
    inside = w.root / "results/tmp-for-verify"
    inside.mkdir()
    before = tree_listing(w.root)
    code, report = run_cli(w, "verify", env_extra={"TMPDIR": str(inside)})
    assert tree_listing(w.root) == before, "verify created content inside the protected runtime root"
    assert code == 2 and "TMPDIR resolves inside" in report["error"]
