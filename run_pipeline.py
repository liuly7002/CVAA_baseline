#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Single entry point for the final CVAA baseline project.

No command-line parameters are used. Edit config.yaml, then run:

    python run_pipeline.py
"""

import sys
from pathlib import Path

from cvaa.config import load_config
from cvaa.pipeline import run_pipeline


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def main() -> None:
    cfg = load_config(CONFIG_PATH)
    summary = run_pipeline(cfg)

    # A non-zero exit code is useful for unattended batch jobs.
    if int(summary.get("routes_failed", 0)) > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
