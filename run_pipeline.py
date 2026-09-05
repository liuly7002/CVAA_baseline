#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CVAA baseline 最终批量处理唯一入口。

用户不需要通过命令行设置数据路径、debug、模型或环境。
所有配置统一写在 config.yaml 中，然后执行：

    conda activate simlingo
    python run_pipeline.py

主流程会自动：
    1. 使用 cvaa_fill 环境启动 LaMa + FLUX worker；
    2. worker 退出并释放显存；
    3. 使用 simlingo 环境启动 Original SimLingo worker；
    4. 计算 AD / FD 并删除当前 chunk 临时文件。
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

    # 若存在失败路线则返回非零状态码，便于无人值守批处理检测失败。
    if int(summary.get("routes_failed", 0)) > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
