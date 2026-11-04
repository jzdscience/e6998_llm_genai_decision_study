#!/usr/bin/env python3
"""Explicit installation/weight download entry point for Linux CUDA machines."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from decision_study.runtime import PROFILES, install_environments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--models", nargs="+", choices=("kev", "startlux", "qwen"), default=["kev", "startlux", "qwen"])
    parser.add_argument("--profile", choices=PROFILES, default=None, help="explicit override; otherwise use saved per-model selections")
    args = parser.parse_args()
    result = install_environments(Path(__file__).resolve().parents[1], args.output_dir, args.models, args.profile)
    print(json.dumps({"status": "installed", "models": list(result["models"]), "environments": list(result["environments"])}))


if __name__ == "__main__":
    main()
