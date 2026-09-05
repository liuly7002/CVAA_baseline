from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import shutil
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import yaml

from .config import discover_routes, resolved_environments, resolved_paths
from .matching import (
    decode_instance_png,
    exact_actor_mask_from_arrays,
    match_route_actors,
)
from .metrics import (
    rank_actor_scores,
    scene_stats,
)



PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKER_DIR = PROJECT_ROOT / "cvaa" / "workers"


def mask_bbox_xyxy(mask_u8: np.ndarray) -> List[int]:
    """返回二值 mask 的 [x1, y1, x2, y2]。"""
    ys, xs = np.nonzero(mask_u8 > 0)
    if len(xs) == 0:
        raise ValueError("Mask is empty.")
    return [
        int(xs.min()),
        int(ys.min()),
        int(xs.max()),
        int(ys.max()),
    ]


def bbox_wh_xyxy(
    bbox_xyxy: List[int],
) -> Tuple[int, int]:
    """根据 xyxy 外接框计算宽和高。"""
    x1, y1, x2, y2 = [
        int(v) for v in bbox_xyxy
    ]
    return (
        int(x2 - x1 + 1),
        int(y2 - y1 + 1),
    )


def _effective_config(
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """移除运行时内部字段，生成可以安全写入 worker JSON 的配置。"""
    return {
        key: value
        for key, value in cfg.items()
        if not str(key).startswith("_")
    }


def _write_json(
    path: Path,
    value: Dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            value,
            f,
            ensure_ascii=False,
            indent=2,
        )


def _read_worker_result(
    path: Path,
    worker_name: str,
) -> Dict[str, Any]:
    if not path.exists():
        raise RuntimeError(
            "%s worker 没有生成结果文件：%s"
            % (worker_name, path)
        )

    payload = json.loads(
        path.read_text(encoding="utf-8")
    )

    if payload.get("status") == "fatal":
        raise RuntimeError(
            "%s worker 失败：%s\n%s"
            % (
                worker_name,
                payload.get("error"),
                payload.get("traceback", ""),
            )
        )

    return payload


def _run_worker(
    python_executable: Path,
    worker_script: Path,
    request_path: Path,
    result_path: Path,
    worker_name: str,
) -> Dict[str, Any]:
    """
    使用指定 Conda 环境的 Python 启动独立 worker。

    这里的 request/result 路径只是主程序与内部 worker 之间的临时 IPC，
    不是用户需要手工填写的命令行参数。
    """
    env = os.environ.copy()

    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(PROJECT_ROOT)
        if not old_pythonpath
        else str(PROJECT_ROOT)
        + os.pathsep
        + old_pythonpath
    )

    cmd = [
        str(python_executable),
        str(worker_script),
        str(request_path),
        str(result_path),
    ]

    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
    )

    if proc.returncode != 0:
        # worker 即使失败也会尽量写 result.json，因此先读取详细错误。
        if result_path.exists():
            return _read_worker_result(
                result_path,
                worker_name,
            )

        raise RuntimeError(
            "%s worker 异常退出，returncode=%d"
            % (worker_name, proc.returncode)
        )

    return _read_worker_result(
        result_path,
        worker_name,
    )


def _preflight_environments(
    cfg: Dict[str, Any],
    envs: Dict[str, Path],
) -> None:
    """
    在正式 actor matching 前检查双环境。

    这样若 cvaa_fill 缺少 FluxFillPipeline，
    会在批处理刚开始时直接报错，而不会等到第一个 chunk 才失败。
    """
    if not bool(
        cfg["environments"].get(
            "validate_on_startup",
            True,
        )
    ):
        return

    print("[ENV] 检查 simlingo 环境：%s" % envs["simlingo_python"])
    sim_code = (
        "import sys, torch, transformers, accelerate; "
        "print('[ENV OK] simlingo python=' + sys.executable); "
        "print('[ENV OK] torch=' + str(torch.__version__) + "
        "' transformers=' + str(transformers.__version__) + "
        "' accelerate=' + str(accelerate.__version__))"
    )
    sim_proc = subprocess.run(
        [
            str(envs["simlingo_python"]),
            "-c",
            sim_code,
        ],
        cwd=str(PROJECT_ROOT),
    )
    if sim_proc.returncode != 0:
        raise RuntimeError(
            "simlingo 环境检查失败。请检查 environments.simlingo_* 配置。"
        )

    print("[ENV] 检查 cvaa_fill 环境：%s" % envs["cvaa_fill_python"])

    backend = str(cfg["inpainting"]["backend"])
    refine = str(
        cfg["inpainting"]["flux_refine_mode"]
    )

    if (
        backend == "flux_fill"
        and refine != "none"
    ):
        fill_code = (
            "import sys, torch, diffusers; "
            "from diffusers import FluxFillPipeline; "
            "print('[ENV OK] cvaa_fill python=' + sys.executable); "
            "print('[ENV OK] torch=' + str(torch.__version__) + "
            "' diffusers=' + str(diffusers.__version__)); "
            "print('[ENV OK] FluxFillPipeline available')"
        )
    else:
        fill_code = (
            "import sys, torch, cv2, numpy, PIL; "
            "print('[ENV OK] cvaa_fill python=' + sys.executable); "
            "print('[ENV OK] basic inpainting dependencies available')"
        )

    fill_proc = subprocess.run(
        [
            str(envs["cvaa_fill_python"]),
            "-c",
            fill_code,
        ],
        cwd=str(PROJECT_ROOT),
    )
    if fill_proc.returncode != 0:
        raise RuntimeError(
            "cvaa_fill 环境检查失败。"
            "请检查 environments.cvaa_fill_* 配置，"
            "并确认 FluxFillPipeline 安装在 cvaa_fill，而不是 simlingo。"
        )


def _json_dump(
    path: Path,
    value: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            value,
            f,
            ensure_ascii=False,
            indent=2,
        )


def _write_jsonl(
    path: Path,
    rows: Iterable[Dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def _read_jsonl(
    path: Path,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(
                    json.loads(line)
                )
    return rows


def _append_jsonl(
    path: Path,
    row: Dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with path.open(
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            json.dumps(
                row,
                ensure_ascii=False,
            )
            + "\n"
        )


def _route_path_hash(
    route_dir: Path,
) -> str:
    return hashlib.sha1(
        str(route_dir.resolve()).encode(
            "utf-8"
        )
    ).hexdigest()[:8]


def build_route_output_map(
    routes: List[Path],
    output_root: Path,
) -> Dict[Path, Path]:
    name_counts: Dict[str, int] = defaultdict(int)
    for route in routes:
        name_counts[route.name] += 1

    mapping: Dict[Path, Path] = {}
    for route in routes:
        name = route.name
        if name_counts[name] > 1:
            name = "%s_%s" % (
                name,
                _route_path_hash(route),
            )
        mapping[route] = (
            output_root / name
        )
    return mapping


def _group_by_frame(
    work_items: List[Dict[str, Any]],
) -> List[List[Dict[str, Any]]]:
    groups: Dict[
        str,
        List[Dict[str, Any]],
    ] = defaultdict(list)
    order: List[str] = []

    for item in work_items:
        frame = str(item["frame"])
        if frame not in groups:
            order.append(frame)
        groups[frame].append(item)

    return [
        groups[frame]
        for frame in order
    ]



def _prepare_route_output(
    route_output: Path,
    cfg: Dict[str, Any],
) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Returns (skip_completed, existing_actor_scores).
    """
    summary_path = (
        route_output / "summary.json"
    )

    if summary_path.exists():
        try:
            summary = json.loads(
                summary_path.read_text(
                    encoding="utf-8"
                )
            )
        except Exception:
            summary = {}

        if (
            summary.get("status")
            == "complete"
            and bool(
                cfg["runtime"][
                    "resume_completed_routes"
                ]
            )
        ):
            print(
                "[RESUME] skip complete route: %s"
                % route_output.name
            )
            return (
                True,
                _read_jsonl(
                    route_output
                    / "actor_scores.jsonl"
                ),
            )

    if route_output.exists() and bool(
        cfg["runtime"][
            "rebuild_incomplete_routes"
        ]
    ):
        shutil.rmtree(
            route_output,
            ignore_errors=True,
        )

    route_output.mkdir(
        parents=True,
        exist_ok=True,
    )
    return False, []



def _write_actor_csv(
    path: Path,
    scores: List[Dict[str, Any]],
) -> None:
    fields = [
        "route_id",
        "frame",
        "rank",
        "actor_id",
        "actor_class",
        "distance_m",
        "match_score",
        "AD",
        "FD",
        "speed_wps_AD",
        "speed_wps_FD",
        "exact_mask_pixels",
        "mask_pixels_used",
        "counterfactual_image",
        "waypoint_debug_image",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()

        for score in scores:
            speed_diag = score.get(
                "speed_wps_diagnostic"
            ) or {}

            writer.writerow(
                {
                    "route_id": score.get(
                        "route_id"
                    ),
                    "frame": score.get(
                        "frame"
                    ),
                    "rank": score.get(
                        "rank"
                    ),
                    "actor_id": score.get(
                        "actor_id"
                    ),
                    "actor_class": score.get(
                        "actor_class"
                    ),
                    "distance_m": score.get(
                        "distance_m"
                    ),
                    "match_score": score.get(
                        "match_score"
                    ),
                    "AD": score.get("AD"),
                    "FD": score.get("FD"),
                    "speed_wps_AD": (
                        speed_diag.get("AD")
                    ),
                    "speed_wps_FD": (
                        speed_diag.get("FD")
                    ),
                    "exact_mask_pixels": (
                        score.get(
                            "exact_mask_pixels"
                        )
                    ),
                    "mask_pixels_used": (
                        score.get(
                            "mask_pixels_used"
                        )
                    ),
                    "counterfactual_image": (
                        score.get(
                            "counterfactual_image"
                        )
                    ),
                    "waypoint_debug_image": (
                        score.get(
                            "waypoint_debug_image"
                        )
                    ),
                }
            )


def _route_summary_stats(
    actor_scores: List[Dict[str, Any]],
) -> Dict[str, Any]:
    ad = [
        float(x["AD"])
        for x in actor_scores
    ]
    fd = [
        float(x["FD"])
        for x in actor_scores
    ]
    return {
        "AD": scene_stats(ad),
        "FD": scene_stats(fd),
    }


def process_route(
    route_dir: Path,
    route_output: Path,
    cfg: Dict[str, Any],
    paths: Dict[str, Optional[Path]],
    envs: Dict[str, Path],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    route_start = time.time()
    route_id = route_output.name

    skip, existing_scores = (
        _prepare_route_output(
            route_output,
            cfg,
        )
    )
    if skip:
        summary = json.loads(
            (
                route_output
                / "summary.json"
            ).read_text(
                encoding="utf-8"
            )
        )
        return summary, existing_scores

    print("=" * 80)
    print(
        "CVAA route: %s"
        % route_dir
    )
    print("=" * 80)

    failures_path = (
        route_output
        / "failures.jsonl"
    )
    skipped_path = (
        route_output
        / "skipped_interventions.jsonl"
    )

    work_items, matching_summary = (
        match_route_actors(
            route_dir,
            cfg,
        )
    )

    print(
        "[MATCH] matched %d / %d actor candidates"
        % (
            matching_summary[
                "matched_total"
            ],
            matching_summary[
                "actor_candidates_total"
            ],
        )
    )

    valid_work: List[Dict[str, Any]] = []
    skipped_count = 0

    # Pre-filter before loading large generative models. Decode each instance
    # frame once, validate all matched actors in that frame, then immediately
    # discard the arrays. No mask image is persisted here.
    for frame_items in _group_by_frame(work_items):
        if not frame_items:
            continue

        try:
            semantic, instance16 = decode_instance_png(
                Path(frame_items[0]["instance_path"])
            )
        except Exception as exc:
            for item in frame_items:
                skipped_count += 1
                _append_jsonl(
                    skipped_path,
                    {
                        "frame": item["frame"],
                        "actor_id": item["actor_id"],
                        "reason": "instance decode failed: %r" % exc,
                    },
                )
            continue

        for item in frame_items:
            try:
                exact_mask = exact_actor_mask_from_arrays(
                    item,
                    semantic,
                    instance16,
                )

                pixels = int(np.count_nonzero(exact_mask))
                mask_cfg = cfg["mask"]

                reason = None
                if pixels < int(mask_cfg["min_mask_pixels"]):
                    reason = (
                        "exact_mask_pixels=%d < min_mask_pixels=%d"
                        % (
                            pixels,
                            int(mask_cfg["min_mask_pixels"]),
                        )
                    )
                else:
                    bbox = mask_bbox_xyxy(exact_mask)
                    width, height = bbox_wh_xyxy(bbox)
                    if min(width, height) < int(
                        mask_cfg["min_object_short_side_px"]
                    ):
                        reason = (
                            "bbox_short_side=%d < min_object_short_side_px=%d"
                            % (
                                min(width, height),
                                int(
                                    mask_cfg[
                                        "min_object_short_side_px"
                                    ]
                                ),
                            )
                        )
                    elif pixels < int(
                        mask_cfg["min_exact_mask_pixels"]
                    ):
                        reason = (
                            "exact_mask_pixels=%d < min_exact_mask_pixels=%d"
                            % (
                                pixels,
                                int(
                                    mask_cfg[
                                        "min_exact_mask_pixels"
                                    ]
                                ),
                            )
                        )

                if reason is not None:
                    skipped_count += 1
                    _append_jsonl(
                        skipped_path,
                        {
                            "frame": item["frame"],
                            "actor_id": item["actor_id"],
                            "reason": reason,
                        },
                    )
                    continue

                valid_work.append(item)

            except Exception as exc:
                skipped_count += 1
                _append_jsonl(
                    skipped_path,
                    {
                        "frame": item["frame"],
                        "actor_id": item["actor_id"],
                        "reason": repr(exc),
                    },
                )

    actor_scores: List[
        Dict[str, Any]
    ] = []
    generated_counterfactuals = 0
    inference_failures = 0
    inpainting_failures = 0
    inpainting_debug_counter = 0
    evaluated_counter = 0
    simlingo_source_info = None
    simlingo_config_path = None

    temp_root = paths[
        "temp_root"
    ]
    if (
        temp_root is not None
        and not temp_root.exists()
    ):
        temp_root.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ------------------------------------------------------------------
    # 双环境 route 级加速流程
    #
    # 旧版本为了控制临时磁盘占用，把一条 route 切成多个 chunk：
    #
    #     chunk 1: 加载 FLUX -> 推理 -> 退出 -> 加载 SimLingo -> 推理 -> 退出
    #     chunk 2: 加载 FLUX -> 推理 -> 退出 -> 加载 SimLingo -> 推理 -> 退出
    #     ...
    #
    # 对一条含数百个 actor 的 route，这会反复加载 FLUX 和 2.57GB 左右的
    # SimLingo checkpoint，模型初始化时间会被重复很多次。
    #
    # 当前版本改为：
    #
    #     当前 route
    #         ↓
    #     cvaa_fill worker 启动 1 次
    #         ↓
    #     LaMa / FLUX 加载 1 次
    #         ↓
    #     生成当前 route 的全部临时反事实图
    #         ↓
    #     cvaa_fill worker 退出并释放 GPU
    #         ↓
    #     simlingo worker 启动 1 次
    #         ↓
    #     Original SimLingo / checkpoint 加载 1 次
    #         ↓
    #     对当前 route 的所有原图/反事实图完成推理、AD/FD、debug
    #         ↓
    #     simlingo worker 退出
    #         ↓
    #     TemporaryDirectory 自动删除当前 route 的全部临时文件
    #
    # 这样保持“双环境隔离”和“中间文件不长期保存”两个原则不变，
    # 但把每条 route 的 FLUX / SimLingo 加载次数从 N 个 chunk 降为 1 次。
    # ------------------------------------------------------------------
    fill_worker_launches = 0
    simlingo_worker_launches = 0

    if valid_work:
        with tempfile.TemporaryDirectory(
            prefix="cvaa_route_",
            dir=(
                str(temp_root)
                if temp_root is not None
                else None
            ),
        ) as tmp:
            tmp_dir = Path(tmp)

            print(
                "[ROUTE CACHE] valid actors=%d, temporary directory=%s"
                % (
                    len(valid_work),
                    tmp_dir,
                )
            )

            # ==========================================================
            # Phase A：为整条 route 准备临时 exact mask 和 worker 请求。
            #
            # exact mask 是很小的 PNG，只作为跨 Conda 环境 IPC 使用。
            # route 完成后会与临时反事实图一起自动删除。
            # ==========================================================
            inpaint_request_items: List[
                Dict[str, Any]
            ] = []

            cached_instance_frame = None
            cached_semantic = None
            cached_instance16 = None

            for item in valid_work:
                frame = str(item["frame"])
                actor_id = str(item["actor_id"])

                try:
                    if cached_instance_frame != frame:
                        (
                            cached_semantic,
                            cached_instance16,
                        ) = decode_instance_png(
                            Path(item["instance_path"])
                        )
                        cached_instance_frame = frame

                    assert cached_semantic is not None
                    assert cached_instance16 is not None

                    exact_mask = exact_actor_mask_from_arrays(
                        item,
                        cached_semantic,
                        cached_instance16,
                    )

                    temp_mask_path = (
                        tmp_dir
                        / "exact_masks"
                        / frame
                        / ("actor_%s.png" % actor_id)
                    )
                    temp_mask_path.parent.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    ok = cv2.imwrite(
                        str(temp_mask_path),
                        exact_mask,
                    )
                    if not ok:
                        raise RuntimeError(
                            "无法写入临时 exact mask：%s"
                            % temp_mask_path
                        )

                    # 如果用户明确要求永久保存反事实图，则直接写到 route_output。
                    # 否则写到当前 route 的临时目录，SimLingo 推理结束后自动删除。
                    if bool(
                        cfg["output"][
                            "save_counterfactual_images"
                        ]
                    ):
                        cf_path = (
                            route_output
                            / "counterfactual_images"
                            / frame
                            / ("actor_%s.png" % actor_id)
                        )
                    else:
                        cf_path = (
                            tmp_dir
                            / "counterfactuals"
                            / frame
                            / ("actor_%s.png" % actor_id)
                        )

                    exact_mask_path = None
                    intervention_mask_path = None

                    if bool(
                        cfg["output"]["save_masks"]
                    ):
                        exact_mask_path = (
                            route_output
                            / "masks"
                            / "exact"
                            / frame
                            / ("actor_%s.png" % actor_id)
                        )
                        intervention_mask_path = (
                            route_output
                            / "masks"
                            / "intervention"
                            / frame
                            / ("actor_%s.png" % actor_id)
                        )

                    inpainting_debug_counter += 1
                    diagnostic_path = None

                    if (
                        bool(cfg["debug"]["enabled"])
                        and bool(
                            cfg["debug"][
                                "save_inpainting_diagnostic"
                            ]
                        )
                        and (
                            inpainting_debug_counter
                            % int(
                                cfg["debug"][
                                    "every_n_actors"
                                ]
                            )
                            == 0
                        )
                    ):
                        diagnostic_path = (
                            route_output
                            / "debug"
                            / "inpainting"
                            / (
                                "%s_actor_%s.jpg"
                                % (frame, actor_id)
                            )
                        )

                    inpaint_request_items.append(
                        {
                            **item,
                            "exact_mask_path":
                                str(temp_mask_path),
                            "counterfactual_image":
                                str(cf_path),
                            "diagnostic_path": (
                                str(diagnostic_path)
                                if diagnostic_path
                                is not None
                                else None
                            ),
                            "save_exact_mask_path": (
                                str(exact_mask_path)
                                if exact_mask_path
                                is not None
                                else None
                            ),
                            "save_intervention_mask_path": (
                                str(
                                    intervention_mask_path
                                )
                                if intervention_mask_path
                                is not None
                                else None
                            ),
                        }
                    )

                except Exception as exc:
                    inpainting_failures += 1
                    _append_jsonl(
                        failures_path,
                        {
                            "stage":
                                "prepare_inpainting_worker",
                            "frame": frame,
                            "actor_id": actor_id,
                            "error": repr(exc),
                        },
                    )
                    print(
                        "[PREPARE ERROR] frame=%s actor=%s: %s"
                        % (
                            frame,
                            actor_id,
                            exc,
                        )
                    )

            # ==========================================================
            # Phase B：整条 route 只启动一次 cvaa_fill worker。
            # ==========================================================
            generated_items: List[
                Dict[str, Any]
            ] = []

            if inpaint_request_items:
                print(
                    "[INPAINT PHASE] start one cvaa_fill worker for %d actors"
                    % len(inpaint_request_items)
                )

                inpaint_request_path = (
                    tmp_dir / "inpaint_request.json"
                )
                inpaint_result_path = (
                    tmp_dir / "inpaint_result.json"
                )

                _write_json(
                    inpaint_request_path,
                    {
                        "config":
                            _effective_config(cfg),
                        "lama_model_path":
                            str(paths["lama_model"]),
                        "flux_model":
                            str(
                                cfg["paths"]["flux_model"]
                            ),
                        "items":
                            inpaint_request_items,
                    },
                )

                fill_worker_launches += 1

                inpaint_payload = _run_worker(
                    python_executable=envs[
                        "cvaa_fill_python"
                    ],
                    worker_script=(
                        WORKER_DIR
                        / "inpaint_worker.py"
                    ),
                    request_path=inpaint_request_path,
                    result_path=inpaint_result_path,
                    worker_name="cvaa_fill",
                )

                request_index = {
                    (
                        str(x["frame"]),
                        str(x["actor_id"]),
                    ): x
                    for x in inpaint_request_items
                }

                for result in inpaint_payload.get(
                    "results",
                    [],
                ):
                    frame = str(result["frame"])
                    actor_id = str(
                        result["actor_id"]
                    )
                    key = (
                        frame,
                        actor_id,
                    )
                    original_item = (
                        request_index[key]
                    )

                    if result["status"] == "ok":
                        meta = (
                            result.get("meta")
                            or {}
                        )
                        generated_counterfactuals += 1

                        generated_items.append(
                            {
                                **original_item,
                                **meta,
                                "counterfactual_image":
                                    original_item[
                                        "counterfactual_image"
                                    ],
                            }
                        )

                    elif result["status"] == "skip":
                        skipped_count += 1
                        _append_jsonl(
                            skipped_path,
                            {
                                "frame": frame,
                                "actor_id": actor_id,
                                "reason":
                                    result.get(
                                        "reason"
                                    ),
                            },
                        )

                    else:
                        inpainting_failures += 1
                        _append_jsonl(
                            failures_path,
                            {
                                "stage":
                                    "inpainting",
                                "frame": frame,
                                "actor_id":
                                    actor_id,
                                "error":
                                    result.get(
                                        "error"
                                    ),
                                "traceback":
                                    result.get(
                                        "traceback"
                                    ),
                            },
                        )
                        print(
                            "[INPAINT ERROR] frame=%s actor=%s: %s"
                            % (
                                frame,
                                actor_id,
                                result.get(
                                    "error"
                                ),
                            )
                        )

                print(
                    "[INPAINT PHASE] finished: generated=%d, failed=%d, skipped=%d"
                    % (
                        len(generated_items),
                        inpainting_failures,
                        skipped_count,
                    )
                )

            # ==========================================================
            # Phase C：cvaa_fill 已退出，GPU 已释放。
            # 整条 route 只启动一次 simlingo worker。
            # ==========================================================
            if generated_items:
                print(
                    "[SIMLINGO PHASE] start one simlingo worker for %d actors"
                    % len(generated_items)
                )

                sim_request_path = (
                    tmp_dir / "simlingo_request.json"
                )
                sim_result_path = (
                    tmp_dir / "simlingo_result.json"
                )

                _write_json(
                    sim_request_path,
                    {
                        "config":
                            _effective_config(cfg),
                        "route_dir":
                            str(route_dir),
                        "route_output":
                            str(route_output),
                        "route_id":
                            route_id,
                        "official_simlingo_root":
                            str(
                                paths[
                                    "official_simlingo_root"
                                ]
                            ),
                        "official_simlingo_checkpoint":
                            str(
                                paths[
                                    "official_simlingo_checkpoint"
                                ]
                            ),
                        "official_simlingo_config": (
                            str(
                                paths[
                                    "official_simlingo_config"
                                ]
                            )
                            if paths[
                                "official_simlingo_config"
                            ]
                            is not None
                            else None
                        ),
                        "evaluated_counter_start":
                            evaluated_counter,
                        "items":
                            generated_items,
                    },
                )

                simlingo_worker_launches += 1

                sim_payload = _run_worker(
                    python_executable=envs[
                        "simlingo_python"
                    ],
                    worker_script=(
                        WORKER_DIR
                        / "simlingo_worker.py"
                    ),
                    request_path=sim_request_path,
                    result_path=sim_result_path,
                    worker_name="simlingo",
                )

                simlingo_source_info = (
                    sim_payload.get(
                        "source_info"
                    )
                )
                simlingo_config_path = (
                    sim_payload.get(
                        "config_path"
                    )
                )

                evaluated_counter = int(
                    sim_payload.get(
                        "evaluated_counter_end",
                        evaluated_counter,
                    )
                )

                new_scores = (
                    sim_payload.get(
                        "scores",
                        [],
                    )
                )
                actor_scores.extend(
                    new_scores
                )

                for failure in (
                    sim_payload.get(
                        "failures",
                        [],
                    )
                ):
                    inference_failures += 1
                    _append_jsonl(
                        failures_path,
                        failure,
                    )
                    print(
                        "[SIMLINGO ERROR] frame=%s actor=%s: %s"
                        % (
                            failure.get(
                                "frame"
                            ),
                            failure.get(
                                "actor_id"
                            ),
                            failure.get(
                                "error"
                            ),
                        )
                    )

                print(
                    "[SIMLINGO PHASE] finished: evaluated=%d, failures=%d"
                    % (
                        len(new_scores),
                        inference_failures,
                    )
                )

            # 离开 TemporaryDirectory 后：
            #   - 临时 exact masks
            #   - 临时 counterfactual images
            #   - worker request/result JSON
            # 全部一次性删除。
            #
            # 若 output.save_counterfactual_images / save_masks 为 true，
            # 对应用户明确要求永久保存的文件不在该临时目录中，不会被删除。

    frame_rankings = rank_actor_scores(
        actor_scores
    )

    actor_scores.sort(
        key=lambda x: (
            str(x["frame"]),
            int(x.get("rank") or 999999),
            str(x["actor_id"]),
        )
    )

    _write_jsonl(
        route_output
        / "actor_scores.jsonl",
        actor_scores,
    )
    _write_actor_csv(
        route_output
        / "actor_scores.csv",
        actor_scores,
    )
    _json_dump(
        route_output
        / "frame_rankings.json",
        frame_rankings,
    )

    summary = {
        "status": "complete",
        "route_id": route_id,
        "route_dir": str(route_dir),
        "elapsed_seconds": (
            time.time() - route_start
        ),
        "matching": matching_summary,
        "matched_work_items": len(
            work_items
        ),
        "valid_interventions_after_mask_filter": len(
            valid_work
        ),
        "execution_strategy": {
            "unit": "route",
            "cvaa_fill_worker_launches": (
                fill_worker_launches
            ),
            "simlingo_worker_launches": (
                simlingo_worker_launches
            ),
            "models_loaded_once_per_route": True,
        },
        "generated_counterfactuals": (
            generated_counterfactuals
        ),
        "evaluated_actor_interventions": len(
            actor_scores
        ),
        "skipped_interventions": (
            skipped_count
        ),
        "inpainting_failures": (
            inpainting_failures
        ),
        "inference_failures": (
            inference_failures
        ),
        "num_ranked_frames": len(
            frame_rankings
        ),
        "formal_ranking": {
            "metric": "pred_route",
            "primary": "AD descending",
            "tie_break": "FD descending",
            "uses_ground_truth": False,
        },
        "result_statistics": (
            _route_summary_stats(
                actor_scores
            )
        ),
        "storage_policy": {
            "intermediate_masks_persisted": bool(
                cfg["output"][
                    "save_masks"
                ]
            ),
            "counterfactual_images_persisted": bool(
                cfg["output"][
                    "save_counterfactual_images"
                ]
            ),
            "trajectories_persisted": bool(
                cfg["output"][
                    "save_trajectories"
                ]
            ),
            "temporary_route_cache_deleted": True,
        },
        "official_simlingo": {
            "source_info": (
                simlingo_source_info
            ),
            "checkpoint": str(
                paths[
                    "official_simlingo_checkpoint"
                ]
            ),
            "hydra_config": (
                simlingo_config_path
            ),
        },
    }

    _json_dump(
        route_output
        / "summary.json",
        summary,
    )

    print(
        "[DONE] %s: %d actors, %d ranked frames, %.1fs"
        % (
            route_id,
            len(actor_scores),
            len(frame_rankings),
            summary["elapsed_seconds"],
        )
    )
    return summary, actor_scores


def _write_global_csv(
    output_root: Path,
    all_scores: List[Dict[str, Any]],
) -> None:
    _write_actor_csv(
        output_root
        / "all_actor_scores.csv",
        all_scores,
    )


def run_pipeline(
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    start = time.time()
    paths = resolved_paths(cfg)
    envs = resolved_environments(cfg)
    _preflight_environments(
        cfg,
        envs,
    )

    output_root = paths[
        "output_root"
    ]
    assert output_root is not None
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Archive the exact effective configuration for reproducibility.
    config_snapshot = {
        key: value
        for key, value in cfg.items()
        if not str(key).startswith("_")
    }
    with (
        output_root / "config_used.yaml"
    ).open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            config_snapshot,
            f,
            sort_keys=False,
            allow_unicode=True,
        )

    routes = discover_routes(cfg)
    route_outputs = (
        build_route_output_map(
            routes,
            output_root,
        )
    )

    print("=" * 80)
    print("CVAA final batch pipeline")
    print("=" * 80)
    print("routes: %d" % len(routes))
    print("output: %s" % output_root)
    print(
        "simlingo python: %s"
        % envs["simlingo_python"]
    )
    print(
        "cvaa_fill python: %s"
        % envs["cvaa_fill_python"]
    )
    print(
        "persistent counterfactual images: %s"
        % bool(
            cfg["output"][
                "save_counterfactual_images"
            ]
        )
    )
    print(
        "persistent masks: %s"
        % bool(
            cfg["output"][
                "save_masks"
            ]
        )
    )
    print("=" * 80)

    route_summaries: List[
        Dict[str, Any]
    ] = []
    all_scores: List[
        Dict[str, Any]
    ] = []
    failed_routes: List[
        Dict[str, Any]
    ] = []

    for index, route_dir in enumerate(
        routes,
        1,
    ):
        print(
            "\n[ROUTE %d/%d] %s"
            % (
                index,
                len(routes),
                route_dir,
            )
        )

        route_output = (
            route_outputs[
                route_dir
            ]
        )

        try:
            summary, scores = (
                process_route(
                    route_dir=route_dir,
                    route_output=route_output,
                    cfg=cfg,
                    paths=paths,
                    envs=envs,
                )
            )
            route_summaries.append(
                summary
            )
            all_scores.extend(
                scores
            )

        except Exception as exc:
            failure = {
                "route_dir": str(
                    route_dir
                ),
                "route_output": str(
                    route_output
                ),
                "error": repr(exc),
            }
            failed_routes.append(
                failure
            )
            print(
                "[ROUTE FAILED] %s: %s"
                % (
                    route_dir,
                    exc,
                )
            )

            route_output.mkdir(
                parents=True,
                exist_ok=True,
            )
            _json_dump(
                route_output
                / "summary.json",
                {
                    "status": "failed",
                    **failure,
                },
            )

    global_rankings = (
        rank_actor_scores(
            all_scores
        )
    )

    all_scores.sort(
        key=lambda x: (
            str(x["route_id"]),
            str(x["frame"]),
            int(x.get("rank") or 999999),
            str(x["actor_id"]),
        )
    )

    _write_jsonl(
        output_root
        / "all_actor_scores.jsonl",
        all_scores,
    )
    _write_global_csv(
        output_root,
        all_scores,
    )
    _write_jsonl(
        output_root
        / "all_frame_rankings.jsonl",
        global_rankings,
    )

    run_summary = {
        "status": (
            "complete"
            if not failed_routes
            else "complete_with_failures"
        ),
        "elapsed_seconds": (
            time.time() - start
        ),
        "routes_discovered": len(
            routes
        ),
        "routes_complete": len(
            route_summaries
        ),
        "routes_failed": len(
            failed_routes
        ),
        "failed_routes": (
            failed_routes
        ),
        "total_actor_interventions": len(
            all_scores
        ),
        "total_ranked_frames": len(
            global_rankings
        ),
        "output_root": str(
            output_root
        ),
        "ranking_rule": {
            "primary": "pred_route AD descending",
            "tie_break": "FD descending",
        },
        "worker_environments": {
            "simlingo_python": str(
                envs["simlingo_python"]
            ),
            "cvaa_fill_python": str(
                envs["cvaa_fill_python"]
            ),
            "isolated_processes": True,
        },
        "storage_policy": {
            "streamed_unified_pipeline": True,
            "route_level_temporary_cache": True,
            "temporary_counterfactuals_deleted_after_each_route": True,
            "models_loaded_once_per_route": True,
            "save_counterfactual_images": bool(
                cfg["output"][
                    "save_counterfactual_images"
                ]
            ),
            "save_masks": bool(
                cfg["output"][
                    "save_masks"
                ]
            ),
        },
    }

    _json_dump(
        output_root
        / "run_summary.json",
        run_summary,
    )

    print("=" * 80)
    print("CVAA batch run finished")
    print("=" * 80)
    print(
        "complete routes: %d"
        % len(route_summaries)
    )
    print(
        "failed routes:   %d"
        % len(failed_routes)
    )
    print(
        "actor scores:    %d"
        % len(all_scores)
    )
    print(
        "ranked frames:   %d"
        % len(global_rankings)
    )
    print(
        "output:          %s"
        % output_root
    )
    print("=" * 80)
    return run_summary
