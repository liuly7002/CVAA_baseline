from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .matching import (
    PLANNING_TO_SEMANTIC,
    build_instance_stats,
    decode_instance_png,
    exact_actor_mask_from_arrays,
    get_rgb_path,
    load_camera_spec,
    load_json_gz,
    normalize_actor_class,
    projected_bbox,
    read_actor_id,
    read_position,
    score_pair,
)


def _raw_frame_number(stem: str, fallback: int) -> int:
    try:
        return int(stem)
    except Exception:
        return int(fallback)


def _mask_bbox_xyxy(mask_u8: np.ndarray) -> List[int]:
    ys, xs = np.nonzero(mask_u8 > 0)
    if len(xs) == 0:
        raise ValueError("Mask is empty.")
    return [
        int(xs.min()),
        int(ys.min()),
        int(xs.max()),
        int(ys.max()),
    ]


def _mask_rejection_reason(
    exact_mask: np.ndarray,
    mask_cfg: Dict[str, Any],
) -> Optional[str]:
    pixels = int(np.count_nonzero(exact_mask))
    if pixels < int(mask_cfg["min_mask_pixels"]):
        return (
            "exact_mask_pixels=%d < min_mask_pixels=%d"
            % (pixels, int(mask_cfg["min_mask_pixels"]))
        )

    x1, y1, x2, y2 = _mask_bbox_xyxy(exact_mask)
    short_side = min(x2 - x1 + 1, y2 - y1 + 1)
    if short_side < int(mask_cfg["min_object_short_side_px"]):
        return (
            "bbox_short_side=%d < min_object_short_side_px=%d"
            % (short_side, int(mask_cfg["min_object_short_side_px"]))
        )

    if pixels < int(mask_cfg["min_exact_mask_pixels"]):
        return (
            "exact_mask_pixels=%d < min_exact_mask_pixels=%d"
            % (pixels, int(mask_cfg["min_exact_mask_pixels"]))
        )
    return None


def selected_frame_paths_manifest(
    route_dir: Path,
    stems: Sequence[str],
) -> List[Tuple[str, Path, Path]]:
    """
    Direct manifest lookup. No glob over instance_front.
    """
    rows: List[Tuple[str, Path, Path]] = []
    inst_dir = route_dir / "instance_front"
    box_dir = route_dir / "boxes"

    for stem in stems:
        inst_path = inst_dir / ("%s.png" % stem)
        box_path = box_dir / ("%s.json.gz" % stem)
        if not inst_path.is_file():
            raise FileNotFoundError(
                "Manifest instance frame missing: %s" % inst_path
            )
        if not box_path.is_file():
            raise FileNotFoundError(
                "Manifest boxes frame missing: %s" % box_path
            )
        rows.append((str(stem), inst_path, box_path))
    return rows


def match_route_actors_manifest(
    route_dir: Path,
    stems: Sequence[str],
    cfg: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    """
    Manifest-only actor matching + exact-mask eligibility.

    Differences from the old path are engineering/correctness only:
      * only explicit keyframes are opened;
      * temporal bonus uses RAW frame-number difference;
      * exact-mask eligibility is evaluated during the same instance decode,
        removing the pipeline's second instance decode / prefilter pass.

    The actor candidate universe, Hungarian assignment, matching score, mask
    thresholds and CVAA ranking definition are unchanged.
    """
    matching_cfg = cfg["matching"]
    data_cfg = cfg["data"]
    mask_cfg = cfg["mask"]
    cam = load_camera_spec(route_dir)

    allowed_classes = {"vehicle", "pedestrian"}
    if bool(data_cfg.get("include_static_obstacles", False)):
        allowed_classes |= {
            "traffic_cone",
            "traffic_warning",
            "barrier",
        }

    temporal_memory: Dict[str, Dict[str, Any]] = {}
    per_actor_assignments: Dict[str, List[int]] = defaultdict(list)

    valid_work: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    frames_total = 0
    frames_with_matches = 0
    frames_with_valid = 0
    actor_candidates_total = 0
    matched_total = 0
    valid_total = 0
    rejected_total = 0
    mask_rejected_total = 0

    rows = selected_frame_paths_manifest(route_dir, stems)

    for selected_index, (stem, inst_path, box_path) in enumerate(rows):
        frames_total += 1
        frame_number = _raw_frame_number(stem, selected_index)

        semantic, instance16 = decode_instance_png(inst_path)
        if semantic.shape != (cam["height"], cam["width"]):
            raise RuntimeError(
                "%s: instance image shape %s != metadata %s"
                % (
                    stem,
                    semantic.shape,
                    (cam["height"], cam["width"]),
                )
            )

        boxes = load_json_gz(box_path)
        if not isinstance(boxes, list):
            boxes = []

        instance_stats = build_instance_stats(semantic, instance16)

        actors: List[Dict[str, Any]] = []
        for box in boxes:
            if not isinstance(box, dict):
                continue

            cls = normalize_actor_class(box)
            if cls not in allowed_classes:
                continue

            actor_id = read_actor_id(box)
            if actor_id is None:
                continue

            pos = read_position(box)
            if pos is None or pos[0] <= 0.0:
                continue

            bbox2d = projected_bbox(box, cls, cam)
            if bbox2d is None:
                continue

            x1, y1, x2, y2 = bbox2d
            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            if area < int(matching_cfg["min_projected_area"]):
                continue

            allowed_sem = PLANNING_TO_SEMANTIC.get(cls)
            if not allowed_sem:
                continue

            actors.append(
                {
                    "actor_id": actor_id,
                    "class": cls,
                    "raw_class": box.get("class"),
                    "bbox": bbox2d,
                    "distance_m": float(np.linalg.norm(pos[:2])),
                    "allowed_semantics": set(int(v) for v in allowed_sem),
                }
            )

        actor_candidates_total += len(actors)

        frame_candidate_ids: List[int] = []
        for iid, st in instance_stats.items():
            sems = set(int(v) for v in st["semantic_counts"].keys())
            if any(
                sems & actor["allowed_semantics"]
                for actor in actors
            ):
                frame_candidate_ids.append(int(iid))

        pair_info: Dict[Tuple[int, int], Dict[str, Any]] = {}
        assignments: Dict[int, int] = {}

        if actors and frame_candidate_ids:
            score_matrix = np.full(
                (len(actors), len(frame_candidate_ids)),
                -1e6,
                dtype=np.float64,
            )

            for ai, actor in enumerate(actors):
                mem = temporal_memory.get(str(actor["actor_id"]))

                for ii, instance_id in enumerate(frame_candidate_ids):
                    temporal_same = False
                    if mem is not None:
                        # Critical sparse-keyframe fix:
                        # use raw frame-number distance, not selected-list index.
                        gap = frame_number - int(mem["frame_number"])
                        temporal_same = (
                            0 < gap <= int(matching_cfg["temporal_max_gap"])
                            and int(mem["instance16"]) == int(instance_id)
                        )

                    info = score_pair(
                        semantic=semantic,
                        instance16=instance16,
                        instance_stats=instance_stats,
                        actor_bbox=actor["bbox"],
                        allowed_semantics=actor["allowed_semantics"],
                        instance_id=instance_id,
                        temporal_same=temporal_same,
                        temporal_bonus=float(matching_cfg["temporal_bonus"]),
                    )
                    if info is None:
                        continue
                    if info["overlap_pixels"] < int(
                        matching_cfg["min_overlap_pixels"]
                    ):
                        continue

                    score_matrix[ai, ii] = info["score"]
                    pair_info[(ai, ii)] = info

            row_ind, col_ind = linear_sum_assignment(-score_matrix)
            assignments = {
                int(ai): int(ii)
                for ai, ii in zip(row_ind, col_ind)
                if score_matrix[ai, ii] > -1e5
            }

        frame_matched = 0
        frame_valid = 0

        for ai, actor in enumerate(actors):
            accepted = False

            if ai in assignments:
                ii = assignments[ai]
                instance_id = frame_candidate_ids[ii]
                info = pair_info[(ai, ii)]

                if info["score"] >= float(matching_cfg["min_score"]):
                    accepted = True
                    frame_matched += 1
                    matched_total += 1

                    temporal_memory[str(actor["actor_id"])] = {
                        "instance16": int(instance_id),
                        "frame_number": int(frame_number),
                    }
                    per_actor_assignments[str(actor["actor_id"])].append(
                        int(instance_id)
                    )

                    rgb_path = get_rgb_path(route_dir, stem)
                    if rgb_path is None:
                        rejected_total += 1
                        skipped.append(
                            {
                                "frame": stem,
                                "actor_id": actor["actor_id"],
                                "reason": "source RGB missing",
                            }
                        )
                        continue

                    item = {
                        "frame": stem,
                        "frame_number": int(frame_number),
                        "actor_id": actor["actor_id"],
                        "actor_class": actor["class"],
                        "raw_class": actor["raw_class"],
                        "distance_m": actor["distance_m"],
                        "instance16": int(instance_id),
                        "match_score": float(info["score"]),
                        "overlap_pixels": int(info["overlap_pixels"]),
                        "containment": float(info["containment"]),
                        "local_dominance": float(info["local_dominance"]),
                        "center_score": float(info["center_score"]),
                        "projected_bbox_xyxy": list(actor["bbox"]),
                        "instance_path": str(inst_path),
                        "boxes_path": str(box_path),
                        "source_image": str(rgb_path),
                    }

                    exact_mask = exact_actor_mask_from_arrays(
                        item, semantic, instance16
                    )
                    reason = _mask_rejection_reason(
                        exact_mask, mask_cfg
                    )
                    if reason is not None:
                        mask_rejected_total += 1
                        skipped.append(
                            {
                                "frame": stem,
                                "actor_id": actor["actor_id"],
                                "reason": reason,
                            }
                        )
                        continue

                    valid_work.append(item)
                    valid_total += 1
                    frame_valid += 1

            if not accepted:
                rejected_total += 1

        if frame_matched > 0:
            frames_with_matches += 1
        if frame_valid > 0:
            frames_with_valid += 1

    temporal_consistencies = []
    for _actor_id, ids in per_actor_assignments.items():
        if len(ids) < 3:
            continue
        counts = Counter(ids)
        _, dominant_n = counts.most_common(1)[0]
        temporal_consistencies.append(
            dominant_n / float(len(ids))
        )

    summary = {
        "allowed_classes": sorted(allowed_classes),
        "frames_total": frames_total,
        "frames_with_matches": frames_with_matches,
        "frames_with_valid_interventions": frames_with_valid,
        "actor_candidates_total": actor_candidates_total,
        "matched_total": matched_total,
        "valid_total": valid_total,
        "rejected_total": rejected_total,
        "mask_rejected_total": mask_rejected_total,
        "match_ratio": (
            float(matched_total / float(actor_candidates_total))
            if actor_candidates_total > 0
            else None
        ),
        "actors_with_matches": len(per_actor_assignments),
        "median_actor_temporal_consistency": (
            float(np.median(temporal_consistencies))
            if temporal_consistencies
            else None
        ),
        "temporal_gap_definition": "raw_frame_number_difference",
    }

    valid_work.sort(
        key=lambda x: (
            int(x.get("frame_number", 0)),
            str(x["actor_id"]),
        )
    )
    return valid_work, summary, skipped
