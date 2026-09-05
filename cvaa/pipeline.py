from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
import shutil
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml

from .config import discover_routes, resolved_paths
from .inpainting import (
    InpaintingEngine,
    SkipIntervention,
    bbox_wh_xyxy,
    mask_bbox_xyxy,
)
from .matching import (
    decode_instance_png,
    exact_actor_mask_from_arrays,
    match_route_actors,
)
from .metrics import (
    compute_metric_pair,
    rank_actor_scores,
    scene_stats,
)
from .simlingo import (
    OfficialSimLingoRunner,
    context_signature,
    load_ground_truth_future_waypoints,
    load_measurement,
    prediction_to_xy,
    save_paired_waypoint_debug,
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


def chunk_work_items(
    work_items: List[Dict[str, Any]],
    max_counterfactuals: int,
) -> List[List[Dict[str, Any]]]:
    """
    Keep every frame intact. A frame containing more actors than the configured
    limit becomes one oversized chunk rather than being split.
    """
    frame_groups = _group_by_frame(
        work_items
    )
    chunks: List[
        List[Dict[str, Any]]
    ] = []
    current: List[
        Dict[str, Any]
    ] = []

    for frame_group in frame_groups:
        if (
            current
            and len(current)
            + len(frame_group)
            > max_counterfactuals
        ):
            chunks.append(current)
            current = []

        current.extend(frame_group)

        if len(current) >= max_counterfactuals:
            chunks.append(current)
            current = []

    if current:
        chunks.append(current)

    return chunks


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

    chunks = chunk_work_items(
        valid_work,
        int(
            cfg["runtime"][
                "max_counterfactuals_per_chunk"
            ]
        ),
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

    for chunk_index, chunk in enumerate(
        chunks,
        1,
    ):
        print(
            "[CHUNK %d/%d] actors=%d"
            % (
                chunk_index,
                len(chunks),
                len(chunk),
            )
        )

        with tempfile.TemporaryDirectory(
            prefix="cvaa_chunk_",
            dir=(
                str(temp_root)
                if temp_root is not None
                else None
            ),
        ) as tmp:
            tmp_dir = Path(tmp)

            # ==============================================================
            # Phase A: counterfactual generation.
            # The large inpainting models are loaded for this phase only.
            # ==============================================================
            inpaint_engine = None
            generated_items: List[
                Dict[str, Any]
            ] = []

            try:
                inpaint_engine = (
                    InpaintingEngine(
                        cfg=cfg,
                        lama_model_path=paths[
                            "lama_model"
                        ],
                        flux_model=str(
                            cfg["paths"][
                                "flux_model"
                            ]
                        ),
                    )
                )

                cached_instance_frame = None
                cached_semantic = None
                cached_instance16 = None

                for item in chunk:
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

                        if bool(
                            cfg["output"][
                                "save_counterfactual_images"
                            ]
                        ):
                            cf_path = (
                                route_output
                                / "counterfactual_images"
                                / frame
                                / (
                                    "actor_%s.png"
                                    % actor_id
                                )
                            )
                        else:
                            cf_path = (
                                tmp_dir
                                / (
                                    "%s_actor_%s.png"
                                    % (
                                        frame,
                                        actor_id,
                                    )
                                )
                            )

                        exact_mask_path = None
                        intervention_mask_path = None
                        if bool(
                            cfg["output"][
                                "save_masks"
                            ]
                        ):
                            exact_mask_path = (
                                route_output
                                / "masks"
                                / "exact"
                                / frame
                                / (
                                    "actor_%s.png"
                                    % actor_id
                                )
                            )
                            intervention_mask_path = (
                                route_output
                                / "masks"
                                / "intervention"
                                / frame
                                / (
                                    "actor_%s.png"
                                    % actor_id
                                )
                            )

                        inpainting_debug_counter += 1
                        diagnostic_path = None
                        if (
                            bool(
                                cfg["debug"][
                                    "enabled"
                                ]
                            )
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
                                    % (
                                        frame,
                                        actor_id,
                                    )
                                )
                            )

                        meta = (
                            inpaint_engine.generate(
                                source_path=Path(
                                    item[
                                        "source_image"
                                    ]
                                ),
                                actor_class=str(
                                    item[
                                        "actor_class"
                                    ]
                                ),
                                exact_mask_u8=(
                                    exact_mask
                                ),
                                output_path=cf_path,
                                diagnostic_path=(
                                    diagnostic_path
                                ),
                                save_exact_mask_path=(
                                    exact_mask_path
                                ),
                                save_intervention_mask_path=(
                                    intervention_mask_path
                                ),
                            )
                        )

                        generated_counterfactuals += 1
                        generated_items.append(
                            {
                                **item,
                                **meta,
                                "counterfactual_image":
                                    str(cf_path),
                            }
                        )

                    except SkipIntervention as exc:
                        skipped_count += 1
                        _append_jsonl(
                            skipped_path,
                            {
                                "frame": frame,
                                "actor_id":
                                    actor_id,
                                "reason":
                                    str(exc),
                            },
                        )

                    except Exception as exc:
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
                                    repr(exc),
                            },
                        )
                        print(
                            "[INPAINT ERROR] frame=%s actor=%s: %s"
                            % (
                                frame,
                                actor_id,
                                exc,
                            )
                        )
            finally:
                if inpaint_engine is not None:
                    inpaint_engine.close()
                    del inpaint_engine
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if not generated_items:
                continue

            # ==============================================================
            # Phase B: original SimLingo paired inference + AD/FD.
            # FLUX/LaMa have already been released, so GPU memory is reused.
            # ==============================================================
            sim_runner = None
            try:
                sim_runner = (
                    OfficialSimLingoRunner(
                        cfg=cfg,
                        official_root=paths[
                            "official_simlingo_root"
                        ],
                        checkpoint=paths[
                            "official_simlingo_checkpoint"
                        ],
                        explicit_config=paths[
                            "official_simlingo_config"
                        ],
                    )
                )
                if simlingo_source_info is None:
                    simlingo_source_info = (
                        sim_runner.source_info
                    )
                    simlingo_config_path = str(
                        sim_runner.config_path
                    )

                items_by_frame = (
                    _group_by_frame(
                        generated_items
                    )
                )

                for frame_items in items_by_frame:
                    frame = str(
                        frame_items[0][
                            "frame"
                        ]
                    )

                    try:
                        measurement, measurement_path = (
                            load_measurement(
                                route_dir,
                                frame,
                            )
                        )

                        (
                            original_prediction,
                            original_context,
                        ) = sim_runner.infer(
                            image_path=Path(
                                frame_items[0][
                                    "source_image"
                                ]
                            ),
                            measurement=measurement,
                        )

                    except Exception as exc:
                        inference_failures += len(
                            frame_items
                        )
                        for item in frame_items:
                            _append_jsonl(
                                failures_path,
                                {
                                    "stage":
                                        "original_inference",
                                    "frame":
                                        frame,
                                    "actor_id":
                                        item[
                                            "actor_id"
                                        ],
                                    "error":
                                        repr(exc),
                                },
                            )
                        print(
                            "[SIMLINGO ERROR] frame=%s original: %s"
                            % (
                                frame,
                                exc,
                            )
                        )
                        continue

                    gt_waypoints = None
                    if (
                        bool(
                            cfg["debug"][
                                "enabled"
                            ]
                        )
                        and bool(
                            cfg["debug"][
                                "save_waypoint_comparison"
                            ]
                        )
                    ):
                        try:
                            original_pred_xy = (
                                prediction_to_xy(
                                    original_prediction.get(
                                        "pred_speed_wps"
                                    ),
                                    "original.pred_speed_wps",
                                )
                            )
                            gt_waypoints = (
                                load_ground_truth_future_waypoints(
                                    route_dir=route_dir,
                                    frame=frame,
                                    num_waypoints=int(
                                        original_pred_xy.shape[
                                            0
                                        ]
                                    ),
                                )
                            )
                        except Exception as exc:
                            print(
                                "[DEBUG WARNING] frame=%s GT: %s"
                                % (
                                    frame,
                                    exc,
                                )
                            )
                            gt_waypoints = None

                    for item in frame_items:
                        actor_id = str(
                            item[
                                "actor_id"
                            ]
                        )
                        try:
                            (
                                cf_prediction,
                                cf_context,
                            ) = sim_runner.infer(
                                image_path=Path(
                                    item[
                                        "counterfactual_image"
                                    ]
                                ),
                                measurement=measurement,
                            )

                            if (
                                context_signature(
                                    original_context
                                )
                                != context_signature(
                                    cf_context
                                )
                            ):
                                raise RuntimeError(
                                    "Non-visual paired-input invariant failed."
                                )

                            route_metric = (
                                compute_metric_pair(
                                    original_prediction.get(
                                        "pred_route"
                                    ),
                                    cf_prediction.get(
                                        "pred_route"
                                    ),
                                    "pred_route",
                                )
                            )

                            speed_diag = None
                            if bool(
                                cfg["output"][
                                    "include_speed_wps_diagnostic"
                                ]
                            ):
                                speed_diag = (
                                    compute_metric_pair(
                                        original_prediction.get(
                                            "pred_speed_wps"
                                        ),
                                        cf_prediction.get(
                                            "pred_speed_wps"
                                        ),
                                        "pred_speed_wps",
                                    )
                                )

                            evaluated_counter += 1
                            waypoint_debug_path = None

                            if (
                                gt_waypoints
                                is not None
                                and bool(
                                    cfg["debug"][
                                        "enabled"
                                    ]
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
                                        % (
                                            frame,
                                            actor_id,
                                        )
                                    )
                                )
                                save_paired_waypoint_debug(
                                    output_path=(
                                        waypoint_debug_path
                                    ),
                                    source_image=Path(
                                        item[
                                            "source_image"
                                        ]
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
                                    gt_waypoints=(
                                        gt_waypoints
                                    ),
                                    frame=frame,
                                    actor_id=actor_id,
                                )

                            final_cf_path = None
                            if bool(
                                cfg["output"][
                                    "save_counterfactual_images"
                                ]
                            ):
                                final_cf_path = str(
                                    item[
                                        "counterfactual_image"
                                    ]
                                )

                            score: Dict[
                                str,
                                Any,
                            ] = {
                                "route_id":
                                    route_id,
                                "route_dir":
                                    str(
                                        route_dir
                                    ),
                                "frame":
                                    frame,
                                "actor_id":
                                    actor_id,
                                "actor_class":
                                    item[
                                        "actor_class"
                                    ],
                                "distance_m":
                                    item.get(
                                        "distance_m"
                                    ),
                                "instance16":
                                    item.get(
                                        "instance16"
                                    ),
                                "match_score":
                                    item.get(
                                        "match_score"
                                    ),
                                "AD":
                                    route_metric[
                                        "AD"
                                    ],
                                "FD":
                                    route_metric[
                                        "FD"
                                    ],
                                "rank":
                                    None,
                                "ranking_trajectory":
                                    "pred_route",
                                "T":
                                    route_metric[
                                        "T"
                                    ],
                                "K_original":
                                    route_metric[
                                        "K_original"
                                    ],
                                "K_counterfactual":
                                    route_metric[
                                        "K_counterfactual"
                                    ],
                                "exact_mask_pixels":
                                    item.get(
                                        "exact_mask_pixels"
                                    ),
                                "mask_pixels_used":
                                    item.get(
                                        "mask_pixels_used"
                                    ),
                                "exact_bbox_xyxy":
                                    item.get(
                                        "exact_bbox_xyxy"
                                    ),
                                "adaptive_dilation_radius_px":
                                    item.get(
                                        "adaptive_dilation_radius_px"
                                    ),
                                "backend":
                                    item.get(
                                        "backend"
                                    ),
                                "pipeline_strategy":
                                    item.get(
                                        "pipeline_strategy"
                                    ),
                                "flux_refine_mode":
                                    item.get(
                                        "flux_refine_mode"
                                    ),
                                "outside_mask_changed_pixels":
                                    item.get(
                                        "outside_mask_changed_pixels"
                                    ),
                                "measurement_path":
                                    str(
                                        measurement_path
                                    ),
                                "source_image":
                                    item.get(
                                        "source_image"
                                    ),
                                "counterfactual_image":
                                    final_cf_path,
                                "waypoint_debug_image":
                                    (
                                        str(
                                            waypoint_debug_path
                                        )
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
                                    "AD":
                                        speed_diag[
                                            "AD"
                                        ],
                                    "FD":
                                        speed_diag[
                                            "FD"
                                        ],
                                    "T":
                                        speed_diag[
                                            "T"
                                        ],
                                }

                            if bool(
                                cfg["simlingo"][
                                    "save_language"
                                ]
                            ):
                                score[
                                    "original_language"
                                ] = (
                                    original_prediction.get(
                                        "language"
                                    )
                                )
                                score[
                                    "counterfactual_language"
                                ] = (
                                    cf_prediction.get(
                                        "language"
                                    )
                                )

                            if bool(
                                cfg["output"][
                                    "save_trajectories"
                                ]
                            ):
                                score[
                                    "original_pred_route"
                                ] = (
                                    route_metric[
                                        "original_mean_trajectory"
                                    ]
                                )
                                score[
                                    "counterfactual_pred_route"
                                ] = (
                                    route_metric[
                                        "counterfactual_mean_trajectory"
                                    ]
                                )
                                score[
                                    "route_displacement_per_timestep"
                                ] = (
                                    route_metric[
                                        "per_timestep_displacement"
                                    ]
                                )
                                if speed_diag is not None:
                                    score[
                                        "speed_wps_diagnostic"
                                    ][
                                        "original_mean_trajectory"
                                    ] = (
                                        speed_diag[
                                            "original_mean_trajectory"
                                        ]
                                    )
                                    score[
                                        "speed_wps_diagnostic"
                                    ][
                                        "counterfactual_mean_trajectory"
                                    ] = (
                                        speed_diag[
                                            "counterfactual_mean_trajectory"
                                        ]
                                    )

                            actor_scores.append(
                                score
                            )

                            progress_every = int(
                                cfg["runtime"][
                                    "progress_every"
                                ]
                            )
                            if (
                                progress_every > 0
                                and evaluated_counter
                                % progress_every
                                == 0
                            ):
                                print(
                                    "[OK] route=%s frame=%s actor=%s AD=%.6f FD=%.6f"
                                    % (
                                        route_id,
                                        frame,
                                        actor_id,
                                        score[
                                            "AD"
                                        ],
                                        score[
                                            "FD"
                                        ],
                                    )
                                )

                        except Exception as exc:
                            inference_failures += 1
                            _append_jsonl(
                                failures_path,
                                {
                                    "stage":
                                        "counterfactual_inference_or_metric",
                                    "frame":
                                        frame,
                                    "actor_id":
                                        actor_id,
                                    "error":
                                        repr(exc),
                                },
                            )
                            print(
                                "[SIMLINGO ERROR] frame=%s actor=%s: %s"
                                % (
                                    frame,
                                    actor_id,
                                    exc,
                                )
                            )

            finally:
                if sim_runner is not None:
                    sim_runner.close()
                    del sim_runner
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # TemporaryDirectory deletes all non-persistent counterfactual PNGs
            # immediately when this chunk exits.

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
        "chunks": len(chunks),
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
            "temporary_chunk_cache_deleted": True,
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
        "storage_policy": {
            "streamed_unified_pipeline": True,
            "bounded_temporary_chunks": True,
            "temporary_counterfactuals_deleted": True,
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
