#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Persistent LaMa + FLUX worker.

The model is loaded ONCE for the whole benchmark run. Jobs are sent as one-line
JSON commands over stdin. Per-route result payloads are written to JSON files,
so large metadata does not travel through stdout.

The parent process pins this worker with CUDA_VISIBLE_DEVICES.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cvaa.inpainting import InpaintingEngine, SkipIntervention  # noqa: E402
from cvaa.matching import (  # noqa: E402
    decode_instance_png,
    exact_actor_mask_from_arrays,
)


READY = "@@CVAA_READY@@"
DONE = "@@CVAA_DONE@@"


def _write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def _group_by_frame(
    items: List[Dict[str, Any]],
) -> List[List[Dict[str, Any]]]:
    groups = defaultdict(list)
    order = []
    for item in items:
        frame = str(item["frame"])
        if frame not in groups:
            order.append(frame)
        groups[frame].append(item)
    return [groups[x] for x in order]


def _process_job(
    engine: InpaintingEngine,
    cfg: Dict[str, Any],
    request_path: Path,
    result_path: Path,
) -> None:
    start = time.perf_counter()
    request = json.loads(request_path.read_text(encoding="utf-8"))
    items: List[Dict[str, Any]] = request["items"]
    results: List[Dict[str, Any]] = []

    progress_every = max(
        1, int(cfg["runtime"].get("progress_every", 50))
    )
    processed = 0

    for frame_items in _group_by_frame(items):
        if not frame_items:
            continue

        semantic, instance16 = decode_instance_png(
            Path(frame_items[0]["instance_path"])
        )

        for item in frame_items:
            processed += 1
            frame = str(item["frame"])
            actor_id = str(item["actor_id"])

            if (
                processed == 1
                or processed == len(items)
                or processed % progress_every == 0
            ):
                print(
                    "[INPAINT %d/%d] frame=%s actor=%s"
                    % (processed, len(items), frame, actor_id),
                    flush=True,
                )

            try:
                exact_mask = exact_actor_mask_from_arrays(
                    item, semantic, instance16
                )

                meta = engine.generate(
                    source_path=Path(item["source_image"]),
                    actor_class=str(item["actor_class"]),
                    exact_mask_u8=exact_mask,
                    output_path=Path(item["counterfactual_image"]),
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
                        if item.get("save_intervention_mask_path")
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

    _write_json(
        result_path,
        {
            "status": "complete",
            "worker": "cvaa_fill_persistent",
            "wall_seconds": float(time.perf_counter() - start),
            "results": results,
        },
    )


def main() -> int:
    if len(sys.argv) != 2:
        print(
            "usage: inpaint_worker_persistent.py <init_request.json>",
            file=sys.stderr,
        )
        return 2

    init_path = Path(sys.argv[1]).expanduser().resolve()
    init = json.loads(init_path.read_text(encoding="utf-8"))
    cfg = init["config"]

    engine = None
    try:
        engine = InpaintingEngine(
            cfg=cfg,
            lama_model_path=Path(init["lama_model_path"]),
            flux_model=str(init["flux_model"]),
        )

        print(READY, flush=True)

        for raw in sys.stdin:
            raw = raw.strip()
            if not raw:
                continue
            command = json.loads(raw)
            op = command.get("op")

            if op == "shutdown":
                break
            if op != "process":
                continue

            job_id = str(command["job_id"])
            request_path = Path(
                command["request_path"]
            ).expanduser().resolve()
            result_path = Path(
                command["result_path"]
            ).expanduser().resolve()

            try:
                _process_job(
                    engine=engine,
                    cfg=cfg,
                    request_path=request_path,
                    result_path=result_path,
                )
            except Exception as exc:
                _write_json(
                    result_path,
                    {
                        "status": "fatal",
                        "worker": "cvaa_fill_persistent",
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    },
                )

            print(DONE + job_id, flush=True)

        return 0

    finally:
        if engine is not None:
            try:
                engine.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
