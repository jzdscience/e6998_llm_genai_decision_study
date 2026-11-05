"""Build, check or serve recorded evidence without starting model services."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = PROJECT / "reports/decision-lab-2026-11-04"
DEFAULT_V1 = PROJECT / "reports/decision-lab-2026-11-02"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="Create a new portable bundle from audited external evidence")
    build.add_argument("--evidence-root", type=Path, required=True)
    build.add_argument("--bundle", type=Path, required=True)
    check = sub.add_parser("check", help="Validate the portable V1/V2/V3 evidence")
    serve = sub.add_parser("serve", help="Launch the CPU viewer on localhost")
    for command in (check, serve):
        command.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
        command.add_argument("--v1-bundle", type=Path, default=DEFAULT_V1)
    serve.add_argument("--port", type=int, default=7860)
    args = parser.parse_args(argv)
    if args.command == "serve" and not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    try:
        if args.command == "build":
            from .evidence import build as build_bundle
            print(build_bundle(args.evidence_root, args.bundle))
        elif args.command == "check":
            from .evidence import load_bundle
            from ..v2.lab.bundle import load_bundle as load_v1
            current = load_bundle(args.bundle)
            original = load_v1(args.v1_bundle)
            print(json.dumps({"status": "verified", "experiments": ["v1", *current["experiments"]],
                              "v1_saved_inputs": len(original["samples"]),
                              "model_calls": 0}, indent=2))
        else:
            from .app import launch
            launch(args.bundle, args.v1_bundle, port=args.port)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"Decision Lab: {exc}\n")
    return 0
