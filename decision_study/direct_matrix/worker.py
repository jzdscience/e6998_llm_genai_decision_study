"""Owned child process for one bounded stage; the parent enforces an absolute watchdog.

    python -m decision_study.direct_matrix.worker --root OUTPUT --attempt FOLDER

Exit 0: the stage finished (``complete.json``). Exit 3: the allocation stop margin
arrived first (``partial.json``; collection resumes later). Exit 2: any other
error (``error.json``). A socket timeout bounds only each read, which is why the
whole stage runs here rather than in the controller.
"""
import argparse
from pathlib import Path
import sys
import traceback

from ..common import StudyError, read_json, utc_now, write_json
from . import collect
from .identity import normal


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--attempt", required=True, type=Path)
    args = parser.parse_args(argv)
    output, folder = normal(args.root), normal(args.attempt)
    if output not in folder.parents:
        print("Attempt folder must be inside the study root", file=sys.stderr)
        return 2
    try:
        plan = read_json(folder / "plan.json")
        if plan.get("kind") == "serial_check":
            collect.serial_check(output, folder, plan)
            return 0
        if plan.get("kind") == "collect":
            from . import study
            context = study.context(output)
            config = study.cell_config_for(output, context, plan["cell_id"])
            collect.collect(output, folder, plan, config)
            return 0
        raise StudyError(f"Unknown stage kind {plan.get('kind')!r}")
    except collect.MarginReached as exc:
        print(f"STOP MARGIN: {exc}", flush=True)
        return 3
    except BaseException as exc:  # noqa: BLE001 - every failure is recorded for the parent
        write_json(folder / "error.json", {"at": utc_now(), "type": type(exc).__name__, "error": str(exc),
                                           "traceback": traceback.format_exc()})
        print(f"STAGE FAILED: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
