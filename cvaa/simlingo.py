from __future__ import annotations

import gc
import gzip
import hashlib
import importlib.util
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoConfig, AutoProcessor


OFFICIAL_REPO = "https://github.com/RenzKa/simlingo"
OFFICIAL_HF_MODEL = "RenzKa/simlingo"

SPECIAL_TOKENS = [
    "<WAYPOINTS>",
    "<WAYPOINTS_DIFF>",
    "<ORG_WAYPOINTS_DIFF>",
    "<ORG_WAYPOINTS>",
    "<WAYPOINT_LAST>",
    "<ROUTE>",
    "<ROUTE_DIFF>",
    "<TARGET_POINT>",
]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")



def git_info(root: Path) -> Dict[str, Optional[str]]:
    out = {"head": None, "remote_origin": None}
    try:
        out["head"] = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        pass
    try:
        out["remote_origin"] = subprocess.check_output(
            ["git", "-C", str(root), "config", "--get", "remote.origin.url"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        pass
    return out



def verify_official_tree(root: Path, allow_nonofficial: bool) -> Dict[str, Any]:
    root = root.expanduser().resolve()

    required = [
        root / "team_code" / "agent_simlingo.py",
        root / "team_code" / "config_simlingo.py",
        root / "team_code" / "simlingo_utils.py",
        root / "simlingo_training" / "models" / "driving.py",
        root / "simlingo_training" / "utils" / "custom_types.py",
        root / "simlingo_training" / "utils" / "internvl2_utils.py",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "The supplied paths.official_simlingo_root is incomplete:\n  "
            + "\n  ".join(missing)
        )

    custom_types = read_text(
        root / "simlingo_training" / "utils" / "custom_types.py"
    )
    driving = read_text(
        root / "simlingo_training" / "models" / "driving.py"
    )

    # These markers exist in the user's modified project but do NOT belong to
    # the released original SimLingo model definition.
    modified_markers = []
    for marker in [
        "counterfactual_prompt",
        "future_interaction_grid",
        "camera_attention_target",
        "counterfactual_waypoints",
    ]:
        if marker in custom_types:
            modified_markers.append(
                f"custom_types.py contains {marker!r}"
            )

    for marker in [
        "FutureInteractionDecoder",
        "PreLanguageInteractionReasoner",
        "target_point_camera_attention",
        "future_interaction_decoder",
    ]:
        if marker in driving:
            modified_markers.append(
                f"driving.py contains {marker!r}"
            )

    info = git_info(root)
    remote = info.get("remote_origin")
    remote_looks_official = (
        remote is None
        or "RenzKa/simlingo" in remote
        or "renzka/simlingo" in remote.lower()
    )

    if modified_markers and not allow_nonofficial:
        raise RuntimeError(
            "Refusing to run because paths.official_simlingo_root appears to be "
            "the MODIFIED network rather than original RenzKa/simlingo:\n  - "
            + "\n  - ".join(modified_markers)
            + "\nUse a clean official checkout. "
              "Only use simlingo.allow_nonofficial_tree for debugging."
        )

    if not remote_looks_official and not allow_nonofficial:
        raise RuntimeError(
            "Git remote does not appear to be RenzKa/simlingo:\n"
            f"  remote.origin.url = {remote}\n"
            "Use a clean official checkout, or simlingo.allow_nonofficial_tree only "
            "for debugging."
        )

    return {
        "root": str(root),
        "git_head": info.get("head"),
        "git_remote_origin": remote,
        "modified_markers_found": modified_markers,
        "official_source_guard_passed": len(modified_markers) == 0,
    }




def activate_official_source_tree(root: Path) -> None:
    """
    Put original RenzKa/simlingo at sys.path[0].

    Repeated calls are allowed as long as already-imported simlingo_training /
    team_code modules originate from the same official source root.
    """
    root = root.expanduser().resolve()
    root_s = str(root)
    sys.path = [p for p in sys.path if p != root_s]
    sys.path.insert(0, root_s)

    polluted = []
    for name, module in list(sys.modules.items()):
        if not (
            name == "simlingo_training"
            or name.startswith("simlingo_training.")
            or name == "team_code"
            or name.startswith("team_code.")
        ):
            continue

        module_file = getattr(module, "__file__", None)
        if module_file is None:
            continue

        try:
            mod_path = Path(module_file).resolve()
            mod_path.relative_to(root)
        except Exception:
            polluted.append("%s -> %s" % (name, module_file))

    if polluted:
        raise RuntimeError(
            "A modified/non-official SimLingo module was imported before the "
            "official source tree. Restart Python. Examples: "
            + "; ".join(polluted[:10])
        )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()



def canonical_frame(value: Any) -> str:
    s = str(value)
    name = Path(s).name
    if name.endswith(".json.gz"):
        name = name[:-8]
    elif name.endswith(".json"):
        name = name[:-5]
    else:
        name = Path(name).stem

    if name.isdigit():
        return f"{int(name):04d}"
    return name



def infer_config_path(
    checkpoint: Path,
    explicit: Optional[Path],
) -> Path:
    if explicit is not None:
        p = explicit.expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(p)
        return p

    # Official agent_simlingo.py:
    # Path(config_path).parent.parent.parent / '.hydra' / 'config.yaml'
    candidate = (
        checkpoint.expanduser().resolve()
        .parent.parent.parent
        / ".hydra"
        / "config.yaml"
    )
    if not candidate.exists():
        raise FileNotFoundError(
            "Could not infer official Hydra config. Expected:\n"
            f"  {candidate}\n"
            "If you downloaded the official Hugging Face model, preserve the "
            "directory structure:\n"
            "  simlingo/.hydra/config.yaml\n"
            "  simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt\n"
            "or set paths.official_simlingo_config in config.yaml."
        )
    return candidate



def load_measurement(
    route_dir: Path,
    frame: str,
) -> Tuple[Dict[str, Any], Path]:
    base = route_dir / "measurements"
    candidates = [
        base / f"{frame}.json.gz",
        base / f"{frame}.json",
    ]

    try:
        raw = str(int(frame))
        candidates.extend(
            [
                base / f"{raw}.json.gz",
                base / f"{raw}.json",
            ]
        )
    except Exception:
        pass

    for path in candidates:
        if not path.exists():
            continue
        if path.name.endswith(".json.gz"):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                return json.load(f), path
        with path.open("r", encoding="utf-8") as f:
            return json.load(f), path

    raise FileNotFoundError(
        f"No measurement for frame={frame}: "
        + ", ".join(str(p) for p in candidates)
    )



def load_ground_truth_future_waypoints(
    route_dir: Path,
    frame: str,
    num_waypoints: int,
) -> np.ndarray:
    """
    Reconstruct SimLingo GT future ego waypoints from saved ego_matrix values.

    This follows the original RenzKa/simlingo BaseDataset.get_waypoints():
        waypoint_ego = R_origin.T @ (p_future - p_origin)

    For N predicted waypoints we load current + N+1 future measurements and
    take [1:-1], matching the official dataset construction.
    """
    if num_waypoints <= 0:
        raise ValueError(f"num_waypoints must be > 0, got {num_waypoints}")

    try:
        frame_index = int(frame)
    except Exception as e:
        raise ValueError(
            f"Debug GT reconstruction requires numeric frame ids, got {frame!r}"
        ) from e

    measurements: List[Dict[str, Any]] = []
    last_measurement: Optional[Dict[str, Any]] = None

    for offset in range(num_waypoints + 2):
        future_frame = f"{frame_index + offset:04d}"
        try:
            measurement, _ = load_measurement(route_dir, future_frame)
            last_measurement = measurement
        except FileNotFoundError:
            if last_measurement is None:
                raise
            # Same behavior as original dataset loader near route end.
            measurement = last_measurement
        measurements.append(measurement)

    origin_matrix = np.asarray(
        measurements[0]["ego_matrix"], dtype=np.float64
    )[:3]
    origin_translation = origin_matrix[:, 3:4]
    origin_rotation = origin_matrix[:, :3]

    waypoints = []
    for idx, measurement in enumerate(measurements):
        if "ego_matrix" not in measurement:
            raise KeyError(
                f"frame={frame}, future_offset={idx}: missing ego_matrix"
            )

        waypoint = np.asarray(
            measurement["ego_matrix"], dtype=np.float64
        )[:3, 3:4]

        waypoint_ego_frame = (
            origin_rotation.T @ (waypoint - origin_translation)
        )
        waypoints.append(waypoint_ego_frame[:2, 0])

    gt = np.asarray(waypoints[1:-1], dtype=np.float32)
    if gt.shape != (num_waypoints, 2):
        raise RuntimeError(
            f"GT shape mismatch: expected {(num_waypoints, 2)}, got {gt.shape}"
        )
    return gt



def prediction_to_xy(value: Any, field_name: str) -> np.ndarray:
    """Normalize prediction to [T,2]."""
    if value is None:
        raise ValueError(f"{field_name} is None")

    arr = np.asarray(value, dtype=np.float32)

    while arr.ndim > 3 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim == 2:
        if arr.shape[-1] < 2:
            raise ValueError(f"{field_name}: invalid shape {arr.shape}")
        return arr[:, :2]

    if arr.ndim == 3:
        if arr.shape[-1] < 2:
            raise ValueError(f"{field_name}: invalid shape {arr.shape}")
        # Current official SimLingo is [1,T,2]. If future K samples exist,
        # display the mean trajectory.
        return arr[..., :2].mean(axis=0)

    raise ValueError(
        f"{field_name}: expected [T,2] or [K,T,2], got {arr.shape}"
    )



def project_ego_waypoints_to_front(
    waypoints_xy: np.ndarray,
    image_width: int,
    image_height: int,
) -> List[Tuple[float, float]]:
    """
    Use original SimLingo's own dataset projection:
        FOV = 110
        get_camera_intrinsics(...)
        project_points(...)
    """
    from simlingo_training.utils.projection import (
        get_camera_intrinsics,
        project_points,
    )

    K = np.asarray(
        get_camera_intrinsics(
            image_width,
            image_height,
            110,
        )
    )

    coords = project_points(waypoints_xy, K)
    return [(float(p[0]), float(p[1])) for p in coords]



def _draw_polyline_and_points(
    image_bgr: np.ndarray,
    points_2d: List[Tuple[float, float]],
    color_bgr: Tuple[int, int, int],
    radius: int,
) -> None:
    h, w = image_bgr.shape[:2]

    valid: List[Optional[Tuple[int, int]]] = []
    for x, y in points_2d:
        if (
            np.isfinite(x)
            and np.isfinite(y)
            and 0 <= x < w
            and 0 <= y < h
        ):
            valid.append((int(round(x)), int(round(y))))
        else:
            valid.append(None)

    for p0, p1 in zip(valid[:-1], valid[1:]):
        if p0 is not None and p1 is not None:
            cv2.line(
                image_bgr,
                p0,
                p1,
                color_bgr,
                2,
                lineType=cv2.LINE_AA,
            )

    for idx, p in enumerate(valid):
        if p is None:
            continue
        cv2.circle(
            image_bgr,
            p,
            radius,
            color_bgr,
            -1,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            image_bgr,
            str(idx + 1),
            (p[0] + radius + 2, p[1] - radius - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color_bgr,
            1,
            cv2.LINE_AA,
        )



def draw_waypoint_debug_panel(
    image_path: Path,
    predicted_waypoints: np.ndarray,
    gt_waypoints: np.ndarray,
    panel_title: str,
) -> np.ndarray:
    """
    RED   = SimLingo pred_speed_wps
    GREEN = GT future ego waypoints
    """
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read debug image: {image_path}")

    h, w = image_bgr.shape[:2]

    pred_2d = project_ego_waypoints_to_front(
        predicted_waypoints, w, h
    )
    gt_2d = project_ego_waypoints_to_front(
        gt_waypoints, w, h
    )

    pred_color = (0, 0, 255)   # red
    gt_color = (0, 255, 0)     # green

    # GT first, prediction second so prediction remains visible when overlapping.
    _draw_polyline_and_points(
        image_bgr, gt_2d, gt_color, radius=5
    )
    _draw_polyline_and_points(
        image_bgr, pred_2d, pred_color, radius=4
    )

    cv2.rectangle(
        image_bgr,
        (8, 8),
        (455, 84),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        image_bgr,
        panel_title,
        (18, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image_bgr,
        "RED: prediction (pred_speed_wps)",
        (18, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        pred_color,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image_bgr,
        "GREEN: ground truth",
        (18, 74),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        gt_color,
        1,
        cv2.LINE_AA,
    )
    return image_bgr



def import_conversation_module(
    official_root: Path,
    vision_variant: str,
) -> Tuple[Any, Path]:
    cache_dir = (
        official_root
        / "pretrained"
        / vision_variant.split("/")[-1]
    )
    model_path = cache_dir / "conversation.py"

    if not model_path.exists():
        from huggingface_hub import snapshot_download

        print(
            f"[OFFICIAL SimLingo] conversation.py missing; downloading "
            f"{vision_variant} -> {cache_dir}"
        )
        snapshot_download(
            repo_id=vision_variant,
            local_dir=str(cache_dir),
        )

    spec = importlib.util.spec_from_file_location(
        "official_simlingo_conv_template",
        str(model_path),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Cannot import conversation.py: {model_path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, cache_dir



def load_official_model(
    official_root: Path,
    checkpoint: Path,
    config_path: Path,
    device: torch.device,
):
    """
    Mirror RenzKa/simlingo team_code/agent_simlingo.py setup().
    """
    cfg = OmegaConf.load(str(config_path))
    cfg.model.vision_model.use_global_img = (
        cfg.data_module.use_global_img
    )

    processor = AutoProcessor.from_pretrained(
        cfg.model.vision_model.variant,
        trust_remote_code=True,
    )
    tokenizer = (
        processor.tokenizer
        if "tokenizer" in processor.__dict__
        else processor
    )
    tokenizer.add_special_tokens(
        {"additional_special_tokens": SPECIAL_TOKENS}
    )
    tokenizer.padding_side = "left"

    cache_dir = (
        official_root
        / "pretrained"
        / str(cfg.model.vision_model.variant).split("/")[-1]
    )

    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = hydra.utils.instantiate(
            cfg.model,
            cfg_data_module=cfg.data_module,
            processor=processor,
            cache_dir=str(cache_dir),
            _recursive_=False,
        ).to(device)
    finally:
        torch.set_default_dtype(default_dtype)

    print(f"[OFFICIAL SimLingo] loading checkpoint: {checkpoint}")
    state_dict = torch.load(
        checkpoint,
        map_location="cpu",
    )

    # Official released pytorch_model.pt is the consolidated state_dict itself.
    if (
        not isinstance(state_dict, dict)
        or len(state_dict) == 0
    ):
        raise RuntimeError(
            "Official pytorch_model.pt did not deserialize to a non-empty dict."
        )

    if "state_dict" in state_dict and isinstance(
        state_dict["state_dict"], dict
    ):
        # Tolerate a Lightning wrapper, but make it explicit.
        print(
            "[WARNING] checkpoint has a state_dict wrapper; "
            "using checkpoint['state_dict']."
        )
        state_dict = state_dict["state_dict"]

    # IMPORTANT: strict=True. We want the official checkpoint to match the
    # official network exactly, not silently fit a modified model.
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    conv_module, _ = import_conversation_module(
        official_root=official_root,
        vision_variant=str(cfg.model.vision_model.variant),
    )

    tmp_config = AutoConfig.from_pretrained(
        cfg.model.vision_model.variant,
        trust_remote_code=True,
    )
    image_size = (
        tmp_config.force_image_size
        or tmp_config.vision_config.image_size
    )
    patch_size = tmp_config.vision_config.patch_size
    num_image_token = int(
        (image_size // patch_size) ** 2
        * (tmp_config.downsample_ratio ** 2)
    )

    return (
        cfg,
        tokenizer,
        model,
        conv_module,
        num_image_token,
    )



def load_rgb_for_official_agent(
    image_path: Path,
    emulate_online_jpeg_roundtrip: bool,
    jpeg_quality: int,
) -> np.ndarray:
    """
    Reproduce the official agent's front-camera image path:

        BGR -> optional JPEG round-trip -> RGB -> bottom crop

    For saved route images, an extra JPEG round-trip is OFF by default because
    the source file already contains dataset JPEG artifacts.
    """
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(
            f"Cannot read image: {image_path}"
        )

    if emulate_online_jpeg_roundtrip:
        ok, encoded = cv2.imencode(
            ".jpg",
            bgr,
            [
                int(cv2.IMWRITE_JPEG_QUALITY),
                int(jpeg_quality),
            ],
        )
        if not ok:
            raise RuntimeError(
                f"JPEG encode failed: {image_path}"
            )
        bgr = cv2.imdecode(
            encoded,
            cv2.IMREAD_UNCHANGED,
        )
        if bgr is None:
            raise RuntimeError(
                f"JPEG decode failed: {image_path}"
            )

    rgb = cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2RGB,
    )

    crop_h = int(
        rgb.shape[0]
        - (rgb.shape[0] * 4.8) // 16
    )
    rgb = rgb[:crop_h, :, :]
    return np.ascontiguousarray(rgb)



def preprocess_official_internvl(
    rgb: np.ndarray,
    cfg: Any,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """
    Mirror the InternVL2 branch in original agent_simlingo.py.
    """
    from simlingo_training.utils.internvl2_utils import (
        build_transform,
        dynamic_preprocess,
    )

    if "internvl2" not in str(
        cfg.model.vision_model.variant
    ).lower():
        raise NotImplementedError(
            "This Stage-3 implementation targets the official SimLingo "
            "InternVL2 inference path."
        )

    transform = build_transform(input_size=448)
    image = Image.fromarray(rgb)

    images = dynamic_preprocess(
        image,
        image_size=448,
        use_thumbnail=bool(
            cfg.model.vision_model.use_global_img
        ),
        max_num=2,
    )
    pixel_values = torch.stack(
        [transform(im) for im in images]
    )
    # Official T=1.
    processed_image = (
        pixel_values
        .unsqueeze(0)
        .unsqueeze(0)
    )

    image_sizes = torch.tensor(
        [[image.size[1], image.size[0]]],
        dtype=torch.int64,
    )

    return (
        processed_image,
        image_sizes,
        int(processed_image.shape[-2]),
        int(processed_image.shape[-1]),
    )



def make_official_navigation_context(
    measurement: Dict[str, Any],
    tokenizer: Any,
) -> Tuple[str, np.ndarray, torch.Tensor, torch.Tensor]:
    """
    The collected measurement already stores ego-frame target_point and
    target_point_next. Reuse them directly, keeping original/counterfactual
    non-visual inputs identical.
    """
    from team_code.config_simlingo import GlobalConfig

    official_cfg = GlobalConfig()

    if official_cfg.eval_route_as not in (
        "target_point",
        "target_point_command",
    ):
        raise NotImplementedError(
            "The released original SimLingo GlobalConfig uses target_point. "
            f"Found eval_route_as={official_cfg.eval_route_as!r}."
        )

    for key in (
        "speed",
        "target_point",
        "target_point_next",
    ):
        if key not in measurement:
            raise KeyError(
                f"measurement missing required field {key!r}"
            )

    speed = float(measurement["speed"])
    speed_rounded = round(speed, 1)

    target_point = np.asarray(
        measurement["target_point"],
        dtype=np.float32,
    )[:2]
    target_point_next = np.asarray(
        measurement["target_point_next"],
        dtype=np.float32,
    )[:2]

    target_points_np = np.stack(
        [target_point, target_point_next],
        axis=0,
    ).astype(np.float32)

    prompt_tp = (
        "Target waypoint: "
        "<TARGET_POINT><TARGET_POINT>."
    )
    if bool(official_cfg.use_cot):
        prompt = (
            f"Current speed: {speed_rounded} m/s. "
            f"{prompt_tp} What should the ego do next?"
        )
    else:
        prompt = (
            f"Current speed: {speed_rounded} m/s. "
            f"{prompt_tp} Predict the waypoints."
        )

    target_point_tensor = torch.from_numpy(
        target_point[None, :]
    ).float()
    speed_tensor = torch.tensor(
        [[speed]],
        dtype=torch.float32,
    )

    return (
        prompt,
        target_points_np,
        target_point_tensor,
        speed_tensor,
    )



def build_official_language_label(
    prompt: str,
    target_points_np: np.ndarray,
    tokenizer: Any,
    conv_module: Any,
    num_image_token: int,
    device: torch.device,
):
    from simlingo_training.utils.custom_types import (
        LanguageLabel,
    )

    # The original agent creates a user turn plus an empty assistant turn.
    conversation = [
        {
            "role": "user",
            "content": prompt,
        },
        {
            "role": "assistant",
            "content": "Waypoints:",
        },
    ]

    template = conv_module.get_conv_template(
        "internlm2-chat"
    )

    for part_idx, part in enumerate(conversation):
        if part["role"] == "assistant":
            template.append_message(
                template.roles[1],
                None,
            )
        elif part["role"] == "user":
            content = part["content"]
            if (
                part_idx == 0
                and "<image>" not in content
            ):
                content = "<image>\n" + content
            template.append_message(
                template.roles[0],
                content,
            )
        else:
            raise ValueError(
                f"Unsupported conversation role: "
                f"{part['role']}"
            )

    query = template.get_prompt()

    system_prompt = (
        template.system_template.replace(
            "{system_message}",
            template.system_message,
        )
        + template.sep
    )
    query = query.replace(
        system_prompt,
        "",
    )

    # Exactly mirror the official agent.
    IMG_START_TOKEN = "<img>"
    IMG_END_TOKEN = "</img>"
    IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
    num_patches_all = 2

    image_tokens = (
        IMG_START_TOKEN
        + IMG_CONTEXT_TOKEN
        * int(num_image_token)
        * num_patches_all
        + IMG_END_TOKEN
    )
    query = query.replace(
        "<image>",
        image_tokens,
        1,
    )

    prompt_batch_list = [query]
    tokenized = tokenizer(
        prompt_batch_list,
        padding=True,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=False,
    )

    phrase_ids = tokenized["input_ids"]
    phrase_valid = (
        phrase_ids != tokenizer.pad_token_id
    )
    phrase_mask = phrase_valid

    target_token_id = int(
        tokenizer.convert_tokens_to_ids(
            "<TARGET_POINT>"
        )
    )

    return LanguageLabel(
        phrase_ids=phrase_ids.to(device),
        phrase_valid=phrase_valid.to(device),
        phrase_mask=phrase_mask.to(device),
        placeholder_values=[
            {
                target_token_id:
                    target_points_np
            }
        ],
        language_string=prompt_batch_list,
        loss_masking=None,
    )



def build_official_driving_input(
    rgb: np.ndarray,
    measurement: Dict[str, Any],
    cfg: Any,
    tokenizer: Any,
    conv_module: Any,
    num_image_token: int,
    device: torch.device,
):
    """
    Construct ORIGINAL RenzKa/simlingo DrivingInput.

    Important: the original DrivingInput has exactly 8 fields and does NOT
    contain counterfactual_prompt or any of the user's later auxiliary fields.
    """
    from simlingo_training.utils.custom_types import (
        DrivingInput,
    )
    from team_code.simlingo_utils import (
        get_camera_extrinsics,
        get_camera_intrinsics,
    )

    expected_fields = (
        "camera_images",
        "image_sizes",
        "camera_intrinsics",
        "camera_extrinsics",
        "vehicle_speed",
        "target_point",
        "prompt",
        "prompt_inference",
    )
    if tuple(DrivingInput._fields) != expected_fields:
        raise RuntimeError(
            "Loaded DrivingInput is NOT the released original SimLingo type.\n"
            f"Expected fields: {expected_fields}\n"
            f"Loaded fields:   {tuple(DrivingInput._fields)}\n"
            "Check paths.official_simlingo_root."
        )

    (
        processed_image,
        image_sizes,
        processed_h,
        processed_w,
    ) = preprocess_official_internvl(
        rgb=rgb,
        cfg=cfg,
    )

    (
        prompt_text,
        target_points_np,
        target_point_tensor,
        speed_tensor,
    ) = make_official_navigation_context(
        measurement=measurement,
        tokenizer=tokenizer,
    )

    ll = build_official_language_label(
        prompt=prompt_text,
        target_points_np=target_points_np,
        tokenizer=tokenizer,
        conv_module=conv_module,
        num_image_token=num_image_token,
        device=device,
    )

    # Preserve the official online agent assignments exactly, including the
    # trailing comma behavior that creates one-element tuples.
    camera_intrinsics = (
        torch.repeat_interleave(
            get_camera_intrinsics(
                processed_w,
                processed_h,
                110,
            ).unsqueeze(0),
            1,
            dim=0,
        )
        .view(1, 3, 3)
        .float()
        .to(device),
    )

    camera_extrinsics = (
        torch.repeat_interleave(
            get_camera_extrinsics().unsqueeze(0),
            1,
            dim=0,
        )
        .view(1, 4, 4)
        .float()
        .to(device),
    )

    driving_input = DrivingInput(
        camera_images=(
            processed_image
            .to(device)
            .bfloat16()
        ),
        image_sizes=image_sizes,
        camera_intrinsics=camera_intrinsics,
        camera_extrinsics=camera_extrinsics,
        vehicle_speed=speed_tensor.to(device),
        target_point=target_point_tensor.to(device),
        prompt=ll,
        prompt_inference=ll,
    )

    context = {
        "speed": float(measurement["speed"]),
        "speed_rounded": round(
            float(measurement["speed"]),
            1,
        ),
        "target_point": (
            np.asarray(
                measurement["target_point"],
                dtype=np.float32,
            )[:2].tolist()
        ),
        "target_point_next": (
            np.asarray(
                measurement["target_point_next"],
                dtype=np.float32,
            )[:2].tolist()
        ),
        "prompt_text": prompt_text,
        "input_rgb_crop_hw": [
            int(rgb.shape[0]),
            int(rgb.shape[1]),
        ],
        "internvl_image_sizes": (
            image_sizes.tolist()
        ),
        "internvl_num_patches": int(
            processed_image.shape[2]
        ),
        "processed_hw": [
            processed_h,
            processed_w,
        ],
    }
    return driving_input, context



def infer_official_simlingo(
    model: torch.nn.Module,
    driving_input: Any,
    save_language: bool,
) -> Dict[str, Any]:
    pred_speed_wps, pred_route, language = model(
        driving_input
    )

    def to_list(x):
        if x is None:
            return None
        return (
            x.detach()
            .float()
            .cpu()
            .numpy()
            .tolist()
        )

    return {
        "pred_route": to_list(pred_route),
        "pred_speed_wps": to_list(
            pred_speed_wps
        ),
        "language": (
            list(language)
            if save_language
            and language is not None
            else None
        ),
    }



def context_signature(
    context: Dict[str, Any],
) -> str:
    keys = [
        "speed",
        "target_point",
        "target_point_next",
        "prompt_text",
        "input_rgb_crop_hw",
        "internvl_image_sizes",
        "internvl_num_patches",
        "processed_hw",
    ]
    obj = {k: context[k] for k in keys}
    payload = json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()




def save_paired_waypoint_debug(
    output_path: Path,
    source_image: Path,
    counterfactual_image: Path,
    original_prediction: Dict[str, Any],
    counterfactual_prediction: Dict[str, Any],
    gt_waypoints: np.ndarray,
    frame: str,
    actor_id: str,
) -> Path:
    """
    LEFT: original RGB + original pred_speed_wps + GT
    RIGHT: counterfactual RGB + counterfactual pred_speed_wps + same GT

    RED = prediction, GREEN = ground truth.
    """
    original_pred = prediction_to_xy(
        original_prediction.get("pred_speed_wps"),
        "original.pred_speed_wps",
    )
    cf_pred = prediction_to_xy(
        counterfactual_prediction.get("pred_speed_wps"),
        "counterfactual.pred_speed_wps",
    )

    if original_pred.shape != gt_waypoints.shape:
        raise RuntimeError(
            "Original prediction / GT shape mismatch: %s vs %s"
            % (original_pred.shape, gt_waypoints.shape)
        )
    if cf_pred.shape != gt_waypoints.shape:
        raise RuntimeError(
            "Counterfactual prediction / GT shape mismatch: %s vs %s"
            % (cf_pred.shape, gt_waypoints.shape)
        )

    left = draw_waypoint_debug_panel(
        source_image,
        original_pred,
        gt_waypoints,
        "Original | frame %s" % frame,
    )
    right = draw_waypoint_debug_panel(
        counterfactual_image,
        cf_pred,
        gt_waypoints,
        "Counterfactual | actor %s" % actor_id,
    )

    if left.shape[0] != right.shape[0]:
        height = min(left.shape[0], right.shape[0])
        left = left[:height]
        right = right[:height]

    gutter = np.zeros(
        (left.shape[0], 12, 3),
        dtype=np.uint8,
    )
    paired = np.concatenate(
        [left, gutter, right],
        axis=1,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(
        str(output_path),
        paired,
        [int(cv2.IMWRITE_JPEG_QUALITY), 95],
    )
    if not ok:
        raise RuntimeError("Failed to save debug image: %s" % output_path)
    return output_path


class OfficialSimLingoRunner(object):
    def __init__(
        self,
        cfg: Dict[str, Any],
        official_root: Path,
        checkpoint: Path,
        explicit_config: Optional[Path],
    ) -> None:
        self.cfg_all = cfg
        self.sim_cfg = cfg["simlingo"]
        self.official_root = official_root.expanduser().resolve()
        self.checkpoint = checkpoint.expanduser().resolve()

        self.source_info = verify_official_tree(
            self.official_root,
            bool(self.sim_cfg.get("allow_nonofficial_tree", False)),
        )
        activate_official_source_tree(self.official_root)

        self.config_path = infer_config_path(
            self.checkpoint,
            explicit_config,
        )
        self.device = torch.device(
            str(self.sim_cfg["device"])
        )
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("SimLingo CUDA requested but unavailable.")

        seed_everything(int(self.sim_cfg["seed"]))

        (
            self.model_cfg,
            self.tokenizer,
            self.model,
            self.conv_module,
            self.num_image_token,
        ) = load_official_model(
            official_root=self.official_root,
            checkpoint=self.checkpoint,
            config_path=self.config_path,
            device=self.device,
        )

    def infer(
        self,
        image_path: Path,
        measurement: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        rgb = load_rgb_for_official_agent(
            image_path=image_path,
            emulate_online_jpeg_roundtrip=bool(
                self.sim_cfg["emulate_online_jpeg_roundtrip"]
            ),
            jpeg_quality=int(self.sim_cfg["jpeg_quality"]),
        )

        driving_input, context = build_official_driving_input(
            rgb=rgb,
            measurement=measurement,
            cfg=self.model_cfg,
            tokenizer=self.tokenizer,
            conv_module=self.conv_module,
            num_image_token=self.num_image_token,
            device=self.device,
        )

        seed_everything(int(self.sim_cfg["seed"]))
        prediction = infer_official_simlingo(
            model=self.model,
            driving_input=driving_input,
            save_language=bool(self.sim_cfg["save_language"]),
        )
        return prediction, context

    def close(self) -> None:
        if hasattr(self, "model"):
            try:
                del self.model
            except Exception:
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
