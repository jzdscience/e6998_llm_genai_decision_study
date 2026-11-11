"""Test-only child process: the real direct-matrix worker plus one injected fault.

    python tests/direct_matrix_fault_worker.py MODE --root OUTPUT --attempt FOLDER

crash-before-receipt  every formal row is persisted, then writing complete.json fails
kill-after-receipt    the worker finishes and writes complete.json, then the process is SIGKILLed
kill-mid-cell         SIGKILL right after the 53rd formal PredictionStore.append: with the default
                      shard size of 25, 50 rows are in shards and 3 only in the fsynced journal.
                      SIGKILL is deliberate; an exception would run predict_examples' finally/sync.
"""
import os
from pathlib import Path
import signal
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decision_study.direct_matrix import collect, worker  # noqa: E402
from decision_study.storage import PredictionStore  # noqa: E402

KILL_AFTER_APPENDS = 53


def main():
    mode, argv = sys.argv[1], sys.argv[2:]
    if mode == "crash-before-receipt":
        real = collect.write_json

        def failing(path, value):
            if Path(path).name == "complete.json":
                raise RuntimeError("injected crash after the final persisted row")
            return real(path, value)

        collect.write_json = failing
        return worker.main(argv)
    if mode == "kill-after-receipt":
        code = worker.main(argv)
        if code == 0:
            os.kill(os.getpid(), signal.SIGKILL)
        return code
    if mode == "kill-mid-cell":
        real, count = PredictionStore.append, [0]

        def append(self, row):
            real(self, row)
            count[0] += 1
            if count[0] == KILL_AFTER_APPENDS:
                os.kill(os.getpid(), signal.SIGKILL)

        PredictionStore.append = append
        return worker.main(argv)
    raise SystemExit(f"unknown fault mode {mode}")


if __name__ == "__main__":
    sys.exit(main())
