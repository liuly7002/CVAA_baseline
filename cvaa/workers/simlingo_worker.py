#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Original SimLingo 独立推理 worker。

本文件不由用户直接运行。
主流程会使用 config.yaml 中解析出的 simlingo Python 自动启动本 worker。

最终加速版中，本 worker 对“一整条 route”只启动一次：

    1. Original SimLingo / checkpoint 只加载一次；
    2. 同一 frame 的 original image 只推理一次；
    3. 每个 actor 的 counterfactual image 分别推理；
    4. 立即计算 pred_route 的 AD / FD；
    5. 可选生成 prediction / GT debug 图；
    6. 当前 route 全部完成后 worker 退出并释放 GPU。

这样避免旧版本按 chunk 反复加载 2GB 以上的 checkpoint。
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


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


def _write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            value,
            f,
            ensure_ascii=False,
            indent=2,
        )


def _group_by_frame(
    items: List[Dict[str, Any]],
) -> List[List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    order: List[str] = []

    for item in items:
        frame = str(item["frame"])
        if frame not in groups:
            order.append(frame)
        groups[frame].append(item)

    return [groups[frame] for frame in order]



def _add_timing(
    accumulator: Dict[str, float],
    timing: Dict[str, Any],
) -> None:
    """累加一次 OfficialSimLingoRunner.infer 的性能字段。"""
    for key in (
        "total_seconds",
        "rgb_load_seconds",
        "input_build_seconds",
        "model_forward_seconds",
    ):
        accumulator[key] = (
            float(accumulator.get(key, 0.0))
            + float(timing.get(key, 0.0) or 0.0)
        )



def main() -> int:
    worker_start = time.perf_counter()

    if len(sys.argv) != 3:
        print(
            "内部 worker 用法错误：simlingo_worker.py <request.json> <result.json>",
            file=sys.stderr,
        )
        return 2

    request_path = Path(sys.argv[1]).expanduser().resolve()
    result_path = Path(sys.argv[2]).expanduser().resolve()

    request = json.loads(
        request_path.read_text(encoding="utf-8")
    )

    cfg = request["config"]
    route_dir = Path(request["route_dir"])
    route_output = Path(request["route_output"])
    route_id = str(request["route_id"])
    items: List[Dict[str, Any]] = request["items"]

    # 该计数器从主进程继承，保证 debug.every_n_actors
    # 跨 chunk 时仍然是全 route 连续计数，而不是每个 chunk 重新从 0 开始。
    evaluated_counter = int(
        request.get("evaluated_counter_start", 0)
    )

    runner = None
    scores: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    original_timing: Dict[str, float] = {}
    counterfactual_timing: Dict[str, float] = {}
    original_inference_count = 0
    counterfactual_inference_count = 0
    measurement_load_seconds = 0.0
    gt_prepare_seconds = 0.0
    metric_seconds = 0.0
    debug_seconds = 0.0

    try:
        print(
            "[SIMLINGO] loading Original SimLingo once for route=%s"
            % route_id,
            flush=True,
        )

        runner_init_start = time.perf_counter()
        runner = OfficialSimLingoRunner(
            cfg=cfg,
            official_root=Path(
                request["official_simlingo_root"]
            ),
            checkpoint=Path(
                request["official_simlingo_checkpoint"]
            ),
            explicit_config=(
                Path(request["official_simlingo_config"])
                if request.get(
                    "official_simlingo_config"
                )
                else None
            ),
        )
        runner_init_seconds = (
            time.perf_counter()
            - runner_init_start
        )

        total_items = len(items)
        processed_items = 0
        progress_every = max(
            1,
            int(
                cfg["runtime"].get(
                    "progress_every",
                    1,
                )
            ),
        )

        for frame_items in _group_by_frame(items):
            frame = str(frame_items[0]["frame"])

            try:
                measurement_start = time.perf_counter()
                measurement, measurement_path = (
                    load_measurement(
                        route_dir,
                        frame,
                    )
                )
                measurement_load_seconds += (
                    time.perf_counter()
                    - measurement_start
                )

                (
                    original_prediction,
                    original_context,
                ) = runner.infer(
                    image_path=Path(
                        frame_items[0]["source_image"]
                    ),
                    measurement=measurement,
                )
                original_inference_count += 1
                _add_timing(
                    original_timing,
                    runner.last_timing,
                )

            except Exception as exc:
                for item in frame_items:
                    failures.append(
                        {
                            "stage": "original_inference",
                            "frame": frame,
                            "actor_id": str(
                                item["actor_id"]
                            ),
                            "error": repr(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )
                continue

            gt_waypoints = None
            gt_start = time.perf_counter()
            if (
                bool(cfg["debug"]["enabled"])
                and bool(
                    cfg["debug"][
                        "save_waypoint_comparison"
                    ]
                )
            ):
                try:
                    original_pred_xy = prediction_to_xy(
                        original_prediction.get(
                            "pred_speed_wps"
                        ),
                        "original.pred_speed_wps",
                    )
                    gt_waypoints = (
                        load_ground_truth_future_waypoints(
                            route_dir=route_dir,
                            frame=frame,
                            num_waypoints=int(
                                original_pred_xy.shape[0]
                            ),
                        )
                    )
                except Exception:
                    gt_waypoints = None
            gt_prepare_seconds += (
                time.perf_counter()
                - gt_start
            )

            for item in frame_items:
                actor_id = str(item["actor_id"])
                processed_items += 1

                if (
                    processed_items == 1
                    or processed_items == total_items
                    or processed_items % progress_every == 0
                ):
                    print(
                        "[SIMLINGO %d/%d] frame=%s actor=%s"
                        % (
                            processed_items,
                            total_items,
                            frame,
                            actor_id,
                        ),
                        flush=True,
                    )

                try:
                    (
                        cf_prediction,
                        cf_context,
                    ) = runner.infer(
                        image_path=Path(
                            item["counterfactual_image"]
                        ),
                        measurement=measurement,
                    )
                    counterfactual_inference_count += 1
                    _add_timing(
                        counterfactual_timing,
                        runner.last_timing,
                    )

                    if (
                        context_signature(
                            original_context
                        )
                        != context_signature(cf_context)
                    ):
                        raise RuntimeError(
                            "非视觉输入一致性检查失败。"
                        )

                    metric_start = time.perf_counter()
                    route_metric = compute_metric_pair(
                        original_prediction.get(
                            "pred_route"
                        ),
                        cf_prediction.get(
                            "pred_route"
                        ),
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

                    metric_seconds += (
                        time.perf_counter()
                        - metric_start
                    )

                    evaluated_counter += 1
                    waypoint_debug_path = None

                    if (
                        gt_waypoints is not None
                        and bool(
                            cfg["debug"]["enabled"]
                        )
                        and bool(
                            cfg["debug"][
                                "save_waypoint_comparison"
                            ]
                        )
                        and (
                            evaluated_counter
                            % int(
                                cfg["debug"][
                                    "every_n_actors"
                                ]
                            )
                            == 0
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

                        debug_start = time.perf_counter()
                        save_paired_waypoint_debug(
                            output_path=waypoint_debug_path,
                            source_image=Path(
                                item["source_image"]
                            ),
                            counterfactual_image=Path(
                                item[
                                    "counterfactual_image"
                                ]
                            ),
                            original_prediction=(
                                original_prediction
                            ),
                            counterfactual_prediction=(
                                cf_prediction
                            ),
                            gt_waypoints=gt_waypoints,
                            frame=frame,
                            actor_id=actor_id,
                        )
                        debug_seconds += (
                            time.perf_counter()
                            - debug_start
                        )

                    final_cf_path = None
                    if bool(
                        cfg["output"][
                            "save_counterfactual_images"
                        ]
                    ):
                        final_cf_path = str(
                            item["counterfactual_image"]
                        )

                    score: Dict[str, Any] = {
                        "route_id": route_id,
                        "route_dir": str(route_dir),
                        "frame": frame,
                        "actor_id": actor_id,
                        "actor_class": item[
                            "actor_class"
                        ],
                        "distance_m": item.get(
                            "distance_m"
                        ),
                        "instance16": item.get(
                            "instance16"
                        ),
                        "match_score": item.get(
                            "match_score"
                        ),
                        "AD": route_metric["AD"],
                        "FD": route_metric["FD"],
                        "rank": None,
                        "ranking_trajectory": (
                            "pred_route"
                        ),
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
                        "adaptive_dilation_radius_px":
                            item.get(
                                "adaptive_dilation_radius_px"
                            ),
                        "backend": item.get("backend"),
                        "pipeline_strategy": item.get(
                            "pipeline_strategy"
                        ),
                        "flux_refine_mode": item.get(
                            "flux_refine_mode"
                        ),
                        "outside_mask_changed_pixels":
                            item.get(
                                "outside_mask_changed_pixels"
                            ),
                        "measurement_path": str(
                            measurement_path
                        ),
                        "source_image": item.get(
                            "source_image"
                        ),
                        "counterfactual_image":
                            final_cf_path,
                        "waypoint_debug_image": (
                            str(waypoint_debug_path)
                            if waypoint_debug_path
                            is not None
                            else None
                        ),
                        "nonvisual_inputs_identical":
                            True,
                    }

                    if speed_diag is not None:
                        score[
                            "speed_wps_diagnostic"
                        ] = {
                            "AD": speed_diag["AD"],
                            "FD": speed_diag["FD"],
                            "T": speed_diag["T"],
                        }

                    if bool(
                        cfg["simlingo"][
                            "save_language"
                        ]
                    ):
                        score[
                            "original_language"
                        ] = original_prediction.get(
                            "language"
                        )
                        score[
                            "counterfactual_language"
                        ] = cf_prediction.get(
                            "language"
                        )

                    if bool(
                        cfg["output"][
                            "save_trajectories"
                        ]
                    ):
                        score[
                            "original_pred_route"
                        ] = route_metric[
                            "original_mean_trajectory"
                        ]
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

                        if speed_diag is not None:
                            score[
                                "speed_wps_diagnostic"
                            ][
                                "original_mean_trajectory"
                            ] = speed_diag[
                                "original_mean_trajectory"
                            ]
                            score[
                                "speed_wps_diagnostic"
                            ][
                                "counterfactual_mean_trajectory"
                            ] = speed_diag[
                                "counterfactual_mean_trajectory"
                            ]

                    scores.append(score)

                    if (
                        processed_items == 1
                        or processed_items == total_items
                        or processed_items % progress_every == 0
                    ):
                        print(
                            "[SIMLINGO OK %d/%d] frame=%s actor=%s AD=%.6f FD=%.6f"
                            % (
                                processed_items,
                                total_items,
                                frame,
                                actor_id,
                                float(score["AD"]),
                                float(score["FD"]),
                            ),
                            flush=True,
                        )

                except Exception as exc:
                    failures.append(
                        {
                            "stage":
                                "counterfactual_inference_or_metric",
                            "frame": frame,
                            "actor_id": actor_id,
                            "error": repr(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )

        import torch
        import transformers
        import accelerate

        original_total = float(
            original_timing.get(
                "total_seconds",
                0.0,
            )
        )
        counterfactual_total = float(
            counterfactual_timing.get(
                "total_seconds",
                0.0,
            )
        )

        performance = {
            "worker_wall_seconds": float(
                time.perf_counter()
                - worker_start
            ),
            "runner_init_seconds": float(
                runner_init_seconds
            ),
            "runner_reported_init_seconds": float(
                getattr(
                    runner,
                    "init_total_seconds",
                    0.0,
                )
            ),
            "source_guard_seconds": float(
                getattr(
                    runner,
                    "source_guard_seconds",
                    0.0,
                )
            ),
            "model_load_seconds": float(
                getattr(
                    runner,
                    "model_load_seconds",
                    0.0,
                )
            ),
            "measurement_load_seconds": float(
                measurement_load_seconds
            ),
            "gt_prepare_seconds": float(
                gt_prepare_seconds
            ),
            "original_inference_count": int(
                original_inference_count
            ),
            "original_inference_total_seconds":
                original_total,
            "original_inference_avg_seconds": (
                original_total
                / original_inference_count
                if original_inference_count > 0
                else 0.0
            ),
            "original_rgb_load_total_seconds": float(
                original_timing.get(
                    "rgb_load_seconds",
                    0.0,
                )
            ),
            "original_input_build_total_seconds": float(
                original_timing.get(
                    "input_build_seconds",
                    0.0,
                )
            ),
            "original_model_forward_total_seconds": float(
                original_timing.get(
                    "model_forward_seconds",
                    0.0,
                )
            ),
            "counterfactual_inference_count": int(
                counterfactual_inference_count
            ),
            "counterfactual_inference_total_seconds":
                counterfactual_total,
            "counterfactual_inference_avg_seconds": (
                counterfactual_total
                / counterfactual_inference_count
                if counterfactual_inference_count > 0
                else 0.0
            ),
            "counterfactual_rgb_load_total_seconds": float(
                counterfactual_timing.get(
                    "rgb_load_seconds",
                    0.0,
                )
            ),
            "counterfactual_input_build_total_seconds": float(
                counterfactual_timing.get(
                    "input_build_seconds",
                    0.0,
                )
            ),
            "counterfactual_model_forward_total_seconds": float(
                counterfactual_timing.get(
                    "model_forward_seconds",
                    0.0,
                )
            ),
            "metric_total_seconds": float(
                metric_seconds
            ),
            "debug_total_seconds": float(
                debug_seconds
            ),
        }

        payload = {
            "status": "complete",
            "worker": "simlingo",
            "python": sys.executable,
            "torch_version": getattr(
                torch, "__version__", None
            ),
            "transformers_version": getattr(
                transformers, "__version__", None
            ),
            "accelerate_version": getattr(
                accelerate, "__version__", None
            ),
            "source_info": runner.source_info,
            "config_path": str(runner.config_path),
            "performance": performance,
            "evaluated_counter_end":
                evaluated_counter,
            "scores": scores,
            "failures": failures,
        }
        print(
            "[PERF][SIMLINGO] worker=%.3fs load=%.3fs "
            "orig=%.3fs (%.3fs/frame) "
            "cf=%.3fs (%.3fs/actor) "
            "model_forward(orig+cf)=%.3fs metric=%.3fs debug=%.3fs"
            % (
                performance[
                    "worker_wall_seconds"
                ],
                performance[
                    "model_load_seconds"
                ],
                performance[
                    "original_inference_total_seconds"
                ],
                performance[
                    "original_inference_avg_seconds"
                ],
                performance[
                    "counterfactual_inference_total_seconds"
                ],
                performance[
                    "counterfactual_inference_avg_seconds"
                ],
                (
                    performance[
                        "original_model_forward_total_seconds"
                    ]
                    + performance[
                        "counterfactual_model_forward_total_seconds"
                    ]
                ),
                performance[
                    "metric_total_seconds"
                ],
                performance[
                    "debug_total_seconds"
                ],
            ),
            flush=True,
        )

        _write_json(result_path, payload)
        return 0

    except Exception as exc:
        _write_json(
            result_path,
            {
                "status": "fatal",
                "worker": "simlingo",
                "python": sys.executable,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            },
        )
        return 1

    finally:
        if runner is not None:
            try:
                runner.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
