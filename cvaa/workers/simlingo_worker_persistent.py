#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Persistent official-SimLingo worker.

Official SimLingo/checkpoint is loaded ONCE for the entire benchmark run.
Original inference is performed once per frame; counterfactual inference is
performed once per valid actor. All forward calls are wrapped in
torch.inference_mode().
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

from cvaa.metrics import compute_metric_pair  # noqa: E402
from cvaa.simlingo import (  # noqa: E402
    OfficialSimLingoRunner,
    context_signature,
    load_ground_truth_future_waypoints,
    load_measurement,
    prediction_to_xy,
    save_paired_waypoint_debug,
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
    runner: OfficialSimLingoRunner,
    cfg: Dict[str, Any],
    request_path: Path,
    result_path: Path,
) -> None:
    import torch

    start = time.perf_counter()
    request = json.loads(request_path.read_text(encoding="utf-8"))

    route_dir = Path(request["route_dir"])
    route_output = Path(request["route_output"])
    route_id = str(request["route_id"])
    items: List[Dict[str, Any]] = request["items"]

    scores: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    progress_every = max(
        1, int(cfg["runtime"].get("progress_every", 50))
    )
    processed_items = 0
    original_count = 0
    cf_count = 0

    for frame_items in _group_by_frame(items):
        if not frame_items:
            continue

        frame = str(frame_items[0]["frame"])

        try:
            measurement, measurement_path = load_measurement(
                route_dir, frame
            )

            with torch.inference_mode():
                original_prediction, original_context = runner.infer(
                    image_path=Path(
                        frame_items[0]["source_image"]
                    ),
                    measurement=measurement,
                )
            original_count += 1

        except Exception as exc:
            for item in frame_items:
                failures.append(
                    {
                        "stage": "original_inference",
                        "frame": frame,
                        "actor_id": str(item["actor_id"]),
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
            continue

        gt_waypoints = None
        if (
            bool(cfg["debug"]["enabled"])
            and bool(
                cfg["debug"]["save_waypoint_comparison"]
            )
        ):
            try:
                original_pred_xy = prediction_to_xy(
                    original_prediction.get("pred_speed_wps"),
                    "original.pred_speed_wps",
                )
                gt_waypoints = load_ground_truth_future_waypoints(
                    route_dir=route_dir,
                    frame=frame,
                    num_waypoints=int(
                        original_pred_xy.shape[0]
                    ),
                )
            except Exception:
                gt_waypoints = None

        for item in frame_items:
            processed_items += 1
            actor_id = str(item["actor_id"])

            if (
                processed_items == 1
                or processed_items == len(items)
                or processed_items % progress_every == 0
            ):
                print(
                    "[SIMLINGO %d/%d] frame=%s actor=%s"
                    % (
                        processed_items,
                        len(items),
                        frame,
                        actor_id,
                    ),
                    flush=True,
                )

            try:
                with torch.inference_mode():
                    cf_prediction, cf_context = runner.infer(
                        image_path=Path(
                            item["counterfactual_image"]
                        ),
                        measurement=measurement,
                    )
                cf_count += 1

                if (
                    context_signature(original_context)
                    != context_signature(cf_context)
                ):
                    raise RuntimeError(
                        "Non-visual input consistency check failed."
                    )

                route_metric = compute_metric_pair(
                    original_prediction.get("pred_route"),
                    cf_prediction.get("pred_route"),
                    "pred_route",
                )

                speed_diag = None
                if bool(
                    cfg["output"][
                        "include_speed_wps_diagnostic"
                    ]
                ):
                    speed_diag = compute_metric_pair(
                        original_prediction.get(
                            "pred_speed_wps"
                        ),
                        cf_prediction.get(
                            "pred_speed_wps"
                        ),
                        "pred_speed_wps",
                    )

                waypoint_debug_path = None
                if (
                    gt_waypoints is not None
                    and bool(cfg["debug"]["enabled"])
                    and bool(
                        cfg["debug"][
                            "save_waypoint_comparison"
                        ]
                    )
                ):
                    waypoint_debug_path = (
                        route_output
                        / "debug"
                        / "waypoints"
                        / (
                            "%s_actor_%s.jpg"
                            % (frame, actor_id)
                        )
                    )
                    save_paired_waypoint_debug(
                        output_path=waypoint_debug_path,
                        source_image=Path(
                            item["source_image"]
                        ),
                        counterfactual_image=Path(
                            item["counterfactual_image"]
                        ),
                        original_prediction=original_prediction,
                        counterfactual_prediction=cf_prediction,
                        gt_waypoints=gt_waypoints,
                        frame=frame,
                        actor_id=actor_id,
                    )

                final_cf_path = (
                    str(item["counterfactual_image"])
                    if bool(
                        cfg["output"][
                            "save_counterfactual_images"
                        ]
                    )
                    else None
                )

                score: Dict[str, Any] = {
                    "route_id": route_id,
                    "route_dir": str(route_dir),
                    "frame": frame,
                    "actor_id": actor_id,
                    "actor_class": item["actor_class"],
                    "distance_m": item.get("distance_m"),
                    "instance16": item.get("instance16"),
                    "match_score": item.get("match_score"),
                    "AD": route_metric["AD"],
                    "FD": route_metric["FD"],
                    "rank": None,
                    "ranking_trajectory": "pred_route",
                    "T": route_metric["T"],
                    "K_original": route_metric[
                        "K_original"
                    ],
                    "K_counterfactual": route_metric[
                        "K_counterfactual"
                    ],
                    "exact_mask_pixels": item.get(
                        "exact_mask_pixels"
                    ),
                    "mask_pixels_used": item.get(
                        "mask_pixels_used"
                    ),
                    "exact_bbox_xyxy": item.get(
                        "exact_bbox_xyxy"
                    ),
                    "adaptive_dilation_radius_px": item.get(
                        "adaptive_dilation_radius_px"
                    ),
                    "backend": item.get("backend"),
                    "pipeline_strategy": item.get(
                        "pipeline_strategy"
                    ),
                    "flux_refine_mode": item.get(
                        "flux_refine_mode"
                    ),
                    "outside_mask_changed_pixels": item.get(
                        "outside_mask_changed_pixels"
                    ),
                    "measurement_path": str(
                        measurement_path
                    ),
                    "source_image": item.get(
                        "source_image"
                    ),
                    "counterfactual_image": final_cf_path,
                    "waypoint_debug_image": (
                        str(waypoint_debug_path)
                        if waypoint_debug_path
                        is not None
                        else None
                    ),
                    "nonvisual_inputs_identical": True,
                }

                if speed_diag is not None:
                    score["speed_wps_diagnostic"] = {
                        "AD": speed_diag["AD"],
                        "FD": speed_diag["FD"],
                        "T": speed_diag["T"],
                    }

                if bool(cfg["simlingo"]["save_language"]):
                    score["original_language"] = (
                        original_prediction.get("language")
                    )
                    score["counterfactual_language"] = (
                        cf_prediction.get("language")
                    )

                if bool(
                    cfg["output"]["save_trajectories"]
                ):
                    score["original_pred_route"] = (
                        route_metric[
                            "original_mean_trajectory"
                        ]
                    )
                    score[
                        "counterfactual_pred_route"
                    ] = route_metric[
                        "counterfactual_mean_trajectory"
                    ]
                    score[
                        "route_displacement_per_timestep"
                    ] = route_metric[
                        "per_timestep_displacement"
                    ]

                scores.append(score)

            except Exception as exc:
                failures.append(
                    {
                        "stage": (
                            "counterfactual_inference_or_metric"
                        ),
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
            "worker": "simlingo_persistent",
            "wall_seconds": float(
                time.perf_counter() - start
            ),
            "original_inference_count": int(
                original_count
            ),
            "counterfactual_inference_count": int(
                cf_count
            ),
            "scores": scores,
            "failures": failures,
        },
    )


def main() -> int:
    if len(sys.argv) != 2:
        print(
            "usage: simlingo_worker_persistent.py <init_request.json>",
            file=sys.stderr,
        )
        return 2

    init_path = Path(sys.argv[1]).expanduser().resolve()
    init = json.loads(init_path.read_text(encoding="utf-8"))
    cfg = init["config"]

    runner = None
    try:
        runner = OfficialSimLingoRunner(
            cfg=cfg,
            official_root=Path(
                init["official_simlingo_root"]
            ),
            checkpoint=Path(
                init["official_simlingo_checkpoint"]
            ),
            explicit_config=(
                Path(init["official_simlingo_config"])
                if init.get(
                    "official_simlingo_config"
                )
                else None
            ),
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
                    runner=runner,
                    cfg=cfg,
                    request_path=request_path,
                    result_path=result_path,
                )
            except Exception as exc:
                _write_json(
                    result_path,
                    {
                        "status": "fatal",
                        "worker": "simlingo_persistent",
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    },
                )

            print(DONE + job_id, flush=True)

        return 0

    finally:
        if runner is not None:
            try:
                runner.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
