from __future__ import annotations

import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


SEMANTIC_NAMES = {
    0: "Unlabeled",
    1: "Roads",
    2: "SideWalks",
    3: "Building",
    4: "Wall",
    5: "Fence",
    6: "Pole",
    7: "TrafficLight",
    8: "TrafficSign",
    9: "Vegetation",
    10: "Terrain",
    11: "Sky",
    12: "Pedestrian",
    13: "Rider",
    14: "Car",
    15: "Truck",
    16: "Bus",
    17: "Train",
    18: "Motorcycle",
    19: "Bicycle",
    20: "Static",
    21: "Dynamic",
    22: "Other",
    23: "Water",
    24: "RoadLine",
    25: "Ground",
    26: "Bridge",
    27: "RailTrack",
    28: "GuardRail",
}

VEHICLE_TAGS = {14, 15, 16, 17, 18, 19}
PEDESTRIAN_TAGS = {12}
STATIC_TAGS = {
    "traffic_cone": {20, 21, 22},
    "traffic_warning": {8, 20, 21, 22},
    "barrier": {5, 20, 21, 22, 28},
}
PLANNING_TO_SEMANTIC = {
    "vehicle": VEHICLE_TAGS,
    "pedestrian": PEDESTRIAN_TAGS,
    **STATIC_TAGS,
}


def load_json_gz(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def load_camera_spec(route_dir: Path) -> Dict[str, Any]:
    path = route_dir / "surround_camera_config.json"
    if not path.exists():
        raise FileNotFoundError("Missing %s" % path)

    with path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    inst = meta.get("front_instance_segmentation")
    if not isinstance(inst, dict):
        raise KeyError(
            "front_instance_segmentation missing in camera metadata: %s" % path
        )

    required = ["position", "rotation", "width", "height", "fov"]
    missing = [k for k in required if k not in inst]
    if missing:
        raise KeyError("Missing camera metadata keys: %s" % missing)

    return {
        "position": np.asarray(inst["position"], dtype=np.float64),
        "rotation": np.asarray(inst["rotation"], dtype=np.float64),
        "width": int(inst["width"]),
        "height": int(inst["height"]),
        "fov": float(inst["fov"]),
    }


def decode_instance_png(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Current collector stores CARLA raw BGRA[:,:,:3] losslessly through OpenCV.

    cv2.imread() returns:
        channel 2 -> CARLA R -> semantic tag
        channel 1 -> CARLA G
        channel 0 -> CARLA B

    image-side 16-bit instance code:
        instance16 = (B << 8) | G
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Failed to read %s" % path)
    if img.ndim != 3 or img.shape[2] < 3:
        raise ValueError("Expected HxWx>=3, got %s" % (img.shape,))
    if img.dtype != np.uint8:
        raise ValueError("Expected uint8, got %s" % img.dtype)

    b = img[:, :, 0].astype(np.uint16)
    g = img[:, :, 1].astype(np.uint16)
    semantic = img[:, :, 2].astype(np.uint8)
    instance16 = (b << 8) | g
    return semantic, instance16


def _raw_box_text(box: Dict[str, Any]) -> str:
    fields = []
    for key in ("class", "type_id", "name", "type"):
        value = box.get(key)
        if value is not None:
            fields.append(str(value).lower())
    return " ".join(fields)


def normalize_actor_class(box: Dict[str, Any]) -> str:
    raw = _raw_box_text(box)
    compact = (
        raw.replace(".", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace(" ", "_")
    )

    if (
        any(k in compact for k in ["ego_car", "ego_info", "ego_vehicle"])
        or compact == "ego"
    ):
        return "ego"

    if any(
        k in compact
        for k in [
            "vehicle",
            "car",
            "truck",
            "bus",
            "van",
            "motorcycle",
            "bicycle",
            "bike",
        ]
    ):
        return "vehicle"

    if any(k in compact for k in ["walker", "pedestrian", "person"]):
        return "pedestrian"

    if any(
        k in compact
        for k in [
            "trafficcone",
            "constructioncone",
            "traffic_cone",
            "construction_cone",
        ]
    ):
        return "traffic_cone"

    if any(
        k in compact
        for k in [
            "trafficwarning",
            "traffic_warning",
            "warningconstruction",
            "warning_construction",
            "constructionwarning",
            "construction_warning",
            "warningaccident",
            "warning_accident",
        ]
    ):
        return "traffic_warning"

    if "barrier" in compact or "barricade" in compact:
        return "barrier"

    return compact


def read_actor_id(box: Dict[str, Any]) -> Optional[Any]:
    value = box.get("id", box.get("track_id"))
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return str(value)


def read_position(box: Dict[str, Any]) -> Optional[np.ndarray]:
    pos = box.get("position")
    if not isinstance(pos, (list, tuple)) or len(pos) < 2:
        return None
    try:
        xyz = [
            float(pos[0]),
            float(pos[1]),
            float(pos[2]) if len(pos) >= 3 else 0.0,
        ]
    except Exception:
        return None
    arr = np.asarray(xyz, dtype=np.float64)
    if not np.isfinite(arr).all():
        return None
    return arr


def read_extent(
    box: Dict[str, Any],
    planning_class: str,
) -> Optional[np.ndarray]:
    ext = box.get("extent")
    if isinstance(ext, (list, tuple)) and len(ext) >= 3:
        try:
            out = np.abs(np.asarray(ext[:3], dtype=np.float64))
        except Exception:
            return None

        raw = _raw_box_text(box).replace(".", "_").replace("-", "_")
        if planning_class == "vehicle" and (
            "static_car" in raw or "staticcar" in raw
        ):
            if out[1] > out[0]:
                out = np.asarray([out[1], out[0], out[2]], dtype=np.float64)
        out = np.maximum(out, 0.05)
        return out

    defaults = {
        "vehicle": [2.25, 1.00, 0.80],
        "pedestrian": [0.35, 0.35, 0.90],
        "traffic_cone": [0.25, 0.25, 0.50],
        "traffic_warning": [0.75, 0.30, 0.75],
        "barrier": [1.00, 0.35, 0.60],
    }
    if planning_class in defaults:
        return np.asarray(defaults[planning_class], dtype=np.float64)
    return None


def read_yaw(box: Dict[str, Any]) -> float:
    try:
        return float(box.get("yaw", 0.0))
    except Exception:
        return 0.0


def carla_rotation_matrix(
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
) -> np.ndarray:
    roll = math.radians(float(roll_deg))
    pitch = math.radians(float(pitch_deg))
    yaw = math.radians(float(yaw_deg))

    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    return np.array(
        [
            [
                cp * cy,
                cy * sp * sr - sy * cr,
                -cy * sp * cr - sy * sr,
            ],
            [
                cp * sy,
                sy * sp * sr + cy * cr,
                -sy * sp * cr + cy * sr,
            ],
            [sp, -cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def camera_intrinsics(
    width: int,
    height: int,
    fov_deg: float,
) -> Tuple[float, float, float, float]:
    f = width / (
        2.0 * math.tan(math.radians(fov_deg) / 2.0)
    )
    return f, f, width / 2.0, height / 2.0


def actor_corners_ego(
    box: Dict[str, Any],
    planning_class: str,
) -> Optional[np.ndarray]:
    center = read_position(box)
    extent = read_extent(box, planning_class)
    if center is None or extent is None:
        return None

    ex, ey, ez = extent
    yaw = read_yaw(box)

    local = np.array(
        [
            [sx * ex, sy * ey, sz * ez]
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ],
        dtype=np.float64,
    )

    c, s = math.cos(yaw), math.sin(yaw)
    rz = np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return local @ rz.T + center[None, :]


def project_points_ego_to_image(
    points_ego: np.ndarray,
    cam: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cam_t = cam["position"]
    roll, pitch, yaw = cam["rotation"]
    r_cam_to_ego = carla_rotation_matrix(roll, pitch, yaw)

    # Row-vector form of ego -> camera local.
    points_cam = (
        points_ego - cam_t[None, :]
    ) @ r_cam_to_ego

    fx, fy, cx, cy = camera_intrinsics(
        cam["width"],
        cam["height"],
        cam["fov"],
    )
    depth = points_cam[:, 0]
    valid = depth > 0.10

    uv = np.full(
        (len(points_ego), 2),
        np.nan,
        dtype=np.float64,
    )
    if np.any(valid):
        x = points_cam[valid, 0]
        y = points_cam[valid, 1]
        z = points_cam[valid, 2]
        uv[valid, 0] = cx + fx * (y / x)
        uv[valid, 1] = cy - fy * (z / x)

    return uv, depth, valid


def projected_bbox(
    box: Dict[str, Any],
    planning_class: str,
    cam: Dict[str, Any],
) -> Optional[Tuple[int, int, int, int]]:
    corners = actor_corners_ego(box, planning_class)
    if corners is None:
        return None

    uv, _, valid = project_points_ego_to_image(corners, cam)
    if np.count_nonzero(valid) < 2:
        return None

    pts = uv[valid]
    if not np.isfinite(pts).all():
        return None

    x1 = int(math.floor(float(np.min(pts[:, 0]))))
    y1 = int(math.floor(float(np.min(pts[:, 1]))))
    x2 = int(math.ceil(float(np.max(pts[:, 0]))))
    y2 = int(math.ceil(float(np.max(pts[:, 1]))))

    w, h = cam["width"], cam["height"]
    if x2 < 0 or y2 < 0 or x1 >= w or y1 >= h:
        return None

    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def build_instance_stats(
    semantic: np.ndarray,
    instance16: np.ndarray,
) -> Dict[int, Dict[str, Any]]:
    stats: Dict[int, Dict[str, Any]] = {}
    ids = np.unique(instance16)
    ids = ids[ids != 0]

    for iid in ids:
        mask = instance16 == iid
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            continue

        sem_values = semantic[mask]
        sem_counts = Counter(int(v) for v in sem_values.tolist())
        stats[int(iid)] = {
            "pixel_count": int(mask.sum()),
            "centroid_x": float(xs.mean()),
            "centroid_y": float(ys.mean()),
            "x1": int(xs.min()),
            "y1": int(ys.min()),
            "x2": int(xs.max()),
            "y2": int(ys.max()),
            "semantic_counts": sem_counts,
        }
    return stats


def score_pair(
    semantic: np.ndarray,
    instance16: np.ndarray,
    instance_stats: Dict[int, Dict[str, Any]],
    actor_bbox: Tuple[int, int, int, int],
    allowed_semantics: Set[int],
    instance_id: int,
    temporal_same: bool,
    temporal_bonus: float,
) -> Optional[Dict[str, Any]]:
    x1, y1, x2, y2 = actor_bbox
    region_sem = semantic[y1 : y2 + 1, x1 : x2 + 1]
    region_inst = instance16[y1 : y2 + 1, x1 : x2 + 1]

    sem_mask = np.isin(region_sem, list(allowed_semantics))
    sem_pixels_in_box = int(np.count_nonzero(sem_mask))
    if sem_pixels_in_box == 0:
        return None

    pair_mask = sem_mask & (region_inst == instance_id)
    overlap = int(np.count_nonzero(pair_mask))
    if overlap == 0:
        return None

    st = instance_stats[instance_id]
    total_instance_pixels = max(int(st["pixel_count"]), 1)

    containment = overlap / float(total_instance_pixels)
    local_dominance = overlap / float(sem_pixels_in_box)

    bx = 0.5 * (x1 + x2)
    by = 0.5 * (y1 + y2)
    bw = max(float(x2 - x1 + 1), 1.0)
    bh = max(float(y2 - y1 + 1), 1.0)
    diag = math.sqrt(bw * bw + bh * bh)
    d = math.sqrt(
        (float(st["centroid_x"]) - bx) ** 2
        + (float(st["centroid_y"]) - by) ** 2
    )
    center_score = max(0.0, 1.0 - d / max(diag, 1.0))

    score = (
        0.50 * containment
        + 0.30 * local_dominance
        + 0.20 * center_score
    )
    if temporal_same:
        score += float(temporal_bonus)

    return {
        "score": float(score),
        "overlap_pixels": overlap,
        "containment": float(containment),
        "local_dominance": float(local_dominance),
        "center_score": float(center_score),
    }


def get_rgb_path(route_dir: Path, stem: str) -> Optional[Path]:
    candidates = [
        route_dir / "rgb_front" / ("%s.jpg" % stem),
        route_dir / "rgb_front" / ("%s.png" % stem),
        route_dir / "rgb" / ("%s.jpg" % stem),
        route_dir / "rgb" / ("%s.png" % stem),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _frame_selected(stem: str, data_cfg: Dict[str, Any]) -> bool:
    try:
        idx = int(stem)
    except Exception:
        idx = None

    start = data_cfg.get("frame_start")
    end = data_cfg.get("frame_end")
    step = int(data_cfg.get("frame_step", 1))

    if idx is not None:
        if start is not None and idx < int(start):
            return False
        if end is not None and idx > int(end):
            return False
        base = int(start) if start is not None else 0
        if (idx - base) % step != 0:
            return False
    return True


def selected_frame_paths(
    route_dir: Path,
    data_cfg: Dict[str, Any],
) -> List[Tuple[str, Path, Path]]:
    inst_dir = route_dir / "instance_front"
    box_dir = route_dir / "boxes"

    rows: List[Tuple[str, Path, Path]] = []
    for inst_path in sorted(inst_dir.glob("*.png")):
        stem = inst_path.stem
        if not _frame_selected(stem, data_cfg):
            continue
        box_path = box_dir / ("%s.json.gz" % stem)
        if box_path.exists():
            rows.append((stem, inst_path, box_path))

    max_frames = int(data_cfg.get("max_frames_per_route", 0) or 0)
    if max_frames > 0:
        rows = rows[:max_frames]
    return rows


def match_route_actors(
    route_dir: Path,
    cfg: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Run actor->instance matching entirely in memory.

    Returns one compact work item per matched actor. No match JSON files or mask
    files are created.
    """
    matching_cfg = cfg["matching"]
    data_cfg = cfg["data"]
    cam = load_camera_spec(route_dir)

    allowed_classes = {"vehicle", "pedestrian"}
    if bool(data_cfg.get("include_static_obstacles", False)):
        allowed_classes |= {"traffic_cone", "traffic_warning", "barrier"}

    temporal_memory: Dict[str, Dict[str, Any]] = {}
    per_actor_assignments: Dict[str, List[int]] = defaultdict(list)

    work_items: List[Dict[str, Any]] = []

    frames_total = 0
    frames_with_matches = 0
    actor_candidates_total = 0
    matched_total = 0
    rejected_total = 0

    for frame_index, (stem, inst_path, box_path) in enumerate(
        selected_frame_paths(route_dir, data_cfg)
    ):
        frames_total += 1
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
                        gap = frame_index - int(mem["frame_index"])
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
                        "frame_index": frame_index,
                    }
                    per_actor_assignments[str(actor["actor_id"])].append(
                        int(instance_id)
                    )

                    rgb_path = get_rgb_path(route_dir, stem)
                    if rgb_path is None:
                        rejected_total += 1
                        continue

                    work_items.append(
                        {
                            "frame": stem,
                            "frame_index": frame_index,
                            "actor_id": actor["actor_id"],
                            "actor_class": actor["class"],
                            "raw_class": actor["raw_class"],
                            "distance_m": actor["distance_m"],
                            "instance16": int(instance_id),
                            "match_score": float(info["score"]),
                            "overlap_pixels": int(info["overlap_pixels"]),
                            "containment": float(info["containment"]),
                            "local_dominance": float(
                                info["local_dominance"]
                            ),
                            "center_score": float(info["center_score"]),
                            "projected_bbox_xyxy": list(actor["bbox"]),
                            "instance_path": str(inst_path),
                            "boxes_path": str(box_path),
                            "source_image": str(rgb_path),
                        }
                    )

            if not accepted:
                rejected_total += 1

        if frame_matched > 0:
            frames_with_matches += 1

    temporal_consistencies = []
    for actor_id, ids in per_actor_assignments.items():
        if len(ids) < 3:
            continue
        counts = Counter(ids)
        _, dominant_n = counts.most_common(1)[0]
        temporal_consistencies.append(dominant_n / float(len(ids)))

    summary = {
        "allowed_classes": sorted(allowed_classes),
        "frames_total": frames_total,
        "frames_with_matches": frames_with_matches,
        "actor_candidates_total": actor_candidates_total,
        "matched_total": matched_total,
        "rejected_total": rejected_total,
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
    }

    work_items.sort(
        key=lambda x: (str(x["frame"]), str(x["actor_id"]))
    )
    return work_items, summary


def exact_actor_mask_from_arrays(
    work_item: Dict[str, Any],
    semantic: np.ndarray,
    instance16: np.ndarray,
) -> np.ndarray:
    """Build exact uint8 0/255 actor mask from already decoded frame arrays."""
    allowed_sem = PLANNING_TO_SEMANTIC.get(
        str(work_item["actor_class"])
    )
    if not allowed_sem:
        raise KeyError(
            "No semantic mapping for actor class %r"
            % work_item["actor_class"]
        )

    exact_bool = (
        (instance16 == int(work_item["instance16"]))
        & np.isin(semantic, list(allowed_sem))
    )
    return exact_bool.astype(np.uint8) * 255


def exact_actor_mask(
    work_item: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return semantic, instance16 and exact actor mask (uint8 0/255).
    """
    semantic, instance16 = decode_instance_png(
        Path(work_item["instance_path"])
    )
    exact_u8 = exact_actor_mask_from_arrays(
        work_item,
        semantic,
        instance16,
    )
    return semantic, instance16, exact_u8
