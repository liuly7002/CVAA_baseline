#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LaMa + FLUX 独立 worker。

本文件不由用户直接运行。
主流程会使用 config.yaml 中解析出的 cvaa_fill Python 自动启动本 worker。

最终加速版中，本 worker 对“一整条 route”只启动一次：

    1. LaMa / FLUX 只加载一次；
    2. 连续生成当前 route 的全部反事实图；
    3. 全部完成后 worker 退出，释放 GPU；
    4. 主流程随后再启动 simlingo worker。

这样既保持双环境隔离，又避免旧版本按 chunk 反复加载 FLUX。
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List

import cv2


# worker 位于 <project>/cvaa/workers/，手动把项目根目录加入 sys.path，
# 保证无论从哪个 Conda 环境启动都能导入当前项目源码。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cvaa.inpainting import InpaintingEngine, SkipIntervention  # noqa: E402


def _write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            value,
            f,
            ensure_ascii=False,
            indent=2,
        )


def main() -> int:
    if len(sys.argv) != 3:
        print(
            "内部 worker 用法错误：inpaint_worker.py <request.json> <result.json>",
            file=sys.stderr,
        )
        return 2

    request_path = Path(sys.argv[1]).expanduser().resolve()
    result_path = Path(sys.argv[2]).expanduser().resolve()

    request = json.loads(
        request_path.read_text(encoding="utf-8")
    )

    cfg = request["config"]
    items: List[Dict[str, Any]] = request["items"]

    engine = None
    results: List[Dict[str, Any]] = []

    try:
        engine = InpaintingEngine(
            cfg=cfg,
            lama_model_path=Path(
                request["lama_model_path"]
            ),
            flux_model=str(request["flux_model"]),
        )

        total_items = len(items)
        progress_every = max(
            1,
            int(
                cfg["runtime"].get(
                    "progress_every",
                    1,
                )
            ),
        )

        for item_index, item in enumerate(
            items,
            1,
        ):
            frame = str(item["frame"])
            actor_id = str(item["actor_id"])

            if (
                item_index == 1
                or item_index == total_items
                or item_index % progress_every == 0
            ):
                print(
                    "[INPAINT %d/%d] frame=%s actor=%s"
                    % (
                        item_index,
                        total_items,
                        frame,
                        actor_id,
                    ),
                    flush=True,
                )

            try:
                exact_mask = cv2.imread(
                    str(item["exact_mask_path"]),
                    cv2.IMREAD_GRAYSCALE,
                )
                if exact_mask is None:
                    raise FileNotFoundError(
                        "无法读取临时 exact mask：%s"
                        % item["exact_mask_path"]
                    )

                meta = engine.generate(
                    source_path=Path(item["source_image"]),
                    actor_class=str(item["actor_class"]),
                    exact_mask_u8=exact_mask,
                    output_path=Path(
                        item["counterfactual_image"]
                    ),
                    diagnostic_path=(
                        Path(item["diagnostic_path"])
                        if item.get("diagnostic_path")
                        else None
                    ),
                    save_exact_mask_path=(
                        Path(item["save_exact_mask_path"])
                        if item.get("save_exact_mask_path")
                        else None
                    ),
                    save_intervention_mask_path=(
                        Path(item["save_intervention_mask_path"])
                        if item.get(
                            "save_intervention_mask_path"
                        )
                        else None
                    ),
                )

                results.append(
                    {
                        "status": "ok",
                        "frame": frame,
                        "actor_id": actor_id,
                        "meta": meta,
                    }
                )

                if (
                    item_index == 1
                    or item_index == total_items
                    or item_index % progress_every == 0
                ):
                    print(
                        "[INPAINT OK %d/%d] frame=%s actor=%s"
                        % (
                            item_index,
                            total_items,
                            frame,
                            actor_id,
                        ),
                        flush=True,
                    )

            except SkipIntervention as exc:
                results.append(
                    {
                        "status": "skip",
                        "frame": frame,
                        "actor_id": actor_id,
                        "reason": str(exc),
                    }
                )

            except Exception as exc:
                results.append(
                    {
                        "status": "error",
                        "frame": frame,
                        "actor_id": actor_id,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                )

        # 记录实际使用的 worker 环境，便于最终论文复现。
        import torch
        import diffusers

        payload = {
            "status": "complete",
            "worker": "cvaa_fill",
            "python": sys.executable,
            "torch_version": getattr(
                torch, "__version__", None
            ),
            "diffusers_version": getattr(
                diffusers, "__version__", None
            ),
            "results": results,
        }
        _write_json(result_path, payload)
        return 0

    except Exception as exc:
        _write_json(
            result_path,
            {
                "status": "fatal",
                "worker": "cvaa_fill",
                "python": sys.executable,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            },
        )
        return 1

    finally:
        if engine is not None:
            try:
                engine.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
