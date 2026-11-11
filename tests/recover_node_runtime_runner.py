"""Run the recovery CLI against a rehearsal world.

usage: python -I -B recover_node_runtime_runner.py SCRIPT PINS PINS_SHA256 {real|stub} COMMAND ARGS...

The rehearsal pins replace the production pins through main()'s keyword arguments (no CLI flag
exists for that). ``stub`` replaces only the two frozen boundaries that need the real node files:
StartLux-9B metadata hashes and direct-matrix plans(), whose inputs embed the original /home paths
and hashes. ``real`` runs both frozen functions unchanged, so on rehearsal data they must fail.
"""
import importlib.util
import sys


def main():
    script, pins, sha, mode, argv = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:]
    spec = importlib.util.spec_from_file_location("recover_node_runtime_under_test", script)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    if mode == "stub":
        original = tool.Matrix.__init__

        def init(self, checkout):
            original(self, checkout)
            self.size._verify_model_metadata = lambda path: None
            self.runtime.plans = lambda dirs, output: ({"source_dirs": dirs, "rehearsal_stub": True}, {}, {})

        tool.Matrix.__init__ = init
    elif mode != "real":
        raise SystemExit(f"unknown mode {mode}")
    raise SystemExit(tool.main(argv, pins_path=pins, pins_sha256=sha))


if __name__ == "__main__":
    main()
