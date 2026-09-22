#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CVAA baseline optimized batch entry.

This entry uses the independent key-frame manifest directly and the persistent
dual-GPU pipeline implemented in cvaa/pipeline_optimized.py.
"""

import sys
from pathlib import Path

from cvaa.config import load_config
from cvaa.pipeline_optimized import run_pipeline


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def main() -> None:
    cfg = load_config(CONFIG_PATH)
    summary = run_pipeline(cfg)
    if int(summary.get("routes_failed", 0)) > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
