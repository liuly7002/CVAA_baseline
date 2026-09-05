from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# 合法 CARLA route 至少应包含这些目录/文件。
# RGB 目录单独判断，因为当前数据可能使用 rgb/ 或 rgb_front/。
REQUIRED_ROUTE_ENTRIES = (
    "instance_front",
    "boxes",
    "measurements",
    "surround_camera_config.json",
)


def load_config(path: Path) -> Dict[str, Any]:
    """读取 YAML 配置并执行完整合法性检查。"""
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError("Missing config file: %s" % path)

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("Top-level config must be a mapping.")

    validate_config(cfg)
    cfg["_config_path"] = str(path)
    return cfg


def _require(cfg: Dict[str, Any], dotted: str) -> Any:
    """读取必须存在的点号配置项，例如 run.input。"""
    cur: Any = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            raise KeyError("Missing required config key: %s" % dotted)
        cur = cur[key]
    return cur


def _as_path(
    value: Any,
    name: str,
    allow_none: bool = False,
) -> Optional[Path]:
    """把配置值转换为绝对路径。"""
    if value is None:
        if allow_none:
            return None
        raise ValueError("%s may not be null" % name)

    return Path(str(value)).expanduser().resolve()


def validate_config(cfg: Dict[str, Any]) -> None:
    """检查配置文件中所有正式运行所需字段。"""
    for section in (
        "run",
        "paths",
        "data",
        "matching",
        "mask",
        "inpainting",
        "simlingo",
        "runtime",
        "debug",
        "output",
    ):
        if section not in cfg or not isinstance(cfg[section], dict):
            raise KeyError("Missing config section: %s" % section)

    # ------------------------------------------------------------------
    # 数据输入
    # ------------------------------------------------------------------
    input_path = _as_path(
        _require(cfg, "run.input"),
        "run.input",
    )
    if input_path is not None and not input_path.exists():
        raise FileNotFoundError(
            "run.input does not exist: %s" % input_path
        )

    recursive = _require(cfg, "run.recursive")
    if not isinstance(recursive, bool):
        raise TypeError("run.recursive must be true or false")

    # recursive=false 时，输入路径本身必须直接就是一条合法 route。
    # recursive=true 时，允许 input 指向任意更高层父目录。
    if not recursive and input_path is not None:
        if not is_route_dir(input_path):
            raise ValueError(
                "run.recursive=false requires run.input to be a valid "
                "CARLA route directory: %s" % input_path
            )

    # ------------------------------------------------------------------
    # 模型与输出路径
    # ------------------------------------------------------------------
    _as_path(
        _require(cfg, "paths.output_root"),
        "paths.output_root",
    )

    official_root = _as_path(
        _require(cfg, "paths.official_simlingo_root"),
        "paths.official_simlingo_root",
    )
    checkpoint = _as_path(
        _require(cfg, "paths.official_simlingo_checkpoint"),
        "paths.official_simlingo_checkpoint",
    )
    lama_model = _as_path(
        _require(cfg, "paths.lama_model"),
        "paths.lama_model",
    )
    explicit_sim_cfg = _as_path(
        cfg["paths"].get("official_simlingo_config"),
        "paths.official_simlingo_config",
        allow_none=True,
    )

    for name, path in (
        ("paths.official_simlingo_root", official_root),
        ("paths.official_simlingo_checkpoint", checkpoint),
        ("paths.lama_model", lama_model),
    ):
        if path is not None and not path.exists():
            raise FileNotFoundError(
                "%s does not exist: %s" % (name, path)
            )

    if explicit_sim_cfg is not None and not explicit_sim_cfg.exists():
        raise FileNotFoundError(
            "paths.official_simlingo_config does not exist: %s"
            % explicit_sim_cfg
        )

    # flux_model 既可以是本地绝对路径，也可以是 Hugging Face model id。
    flux_value = str(_require(cfg, "paths.flux_model"))
    flux_path = Path(flux_value).expanduser()
    if flux_path.is_absolute() and not flux_path.exists():
        raise FileNotFoundError(
            "paths.flux_model local directory does not exist: %s"
            % flux_path
        )

    # ------------------------------------------------------------------
    # Inpainting
    # ------------------------------------------------------------------
    backend = str(_require(cfg, "inpainting.backend"))
    if backend not in ("flux_fill", "lama_only", "opencv"):
        raise ValueError(
            "Unsupported inpainting.backend=%r" % backend
        )

    refine = str(_require(cfg, "inpainting.flux_refine_mode"))
    if refine not in ("seam", "full", "none"):
        raise ValueError(
            "inpainting.flux_refine_mode must be seam/full/none"
        )

    if bool(_require(cfg, "inpainting.cpu_offload")) and bool(
        _require(cfg, "inpainting.sequential_cpu_offload")
    ):
        raise ValueError(
            "Enable only one of inpainting.cpu_offload and "
            "inpainting.sequential_cpu_offload."
        )

    # ------------------------------------------------------------------
    # Runtime / 数据范围
    # ------------------------------------------------------------------
    if int(_require(cfg, "runtime.max_counterfactuals_per_chunk")) <= 0:
        raise ValueError(
            "runtime.max_counterfactuals_per_chunk must be > 0"
        )

    if int(_require(cfg, "data.frame_step")) <= 0:
        raise ValueError("data.frame_step must be > 0")

    if int(_require(cfg, "debug.every_n_actors")) <= 0:
        raise ValueError("debug.every_n_actors must be > 0")

    nonnegative_ints = [
        "matching.min_projected_area",
        "matching.min_overlap_pixels",
        "matching.temporal_max_gap",
        "mask.min_mask_pixels",
        "mask.min_object_short_side_px",
        "mask.min_exact_mask_pixels",
        "mask.adaptive_dilate_min_px",
        "mask.adaptive_dilate_max_px",
        "inpainting.num_inference_steps",
        "inpainting.crop_min_side_px",
        "inpainting.crop_max_side_px",
        "inpainting.crop_target_size",
        "inpainting.flux_seam_width_px",
        "data.max_routes",
        "data.max_frames_per_route",
    ]
    for dotted in nonnegative_ints:
        if int(_require(cfg, dotted)) < 0:
            raise ValueError("%s must be >= 0" % dotted)

    if float(_require(cfg, "matching.min_score")) < 0.0:
        raise ValueError("matching.min_score must be >= 0")

    if float(_require(cfg, "mask.adaptive_dilate_ratio")) < 0.0:
        raise ValueError(
            "mask.adaptive_dilate_ratio must be >= 0"
        )


def is_route_dir(path: Path) -> bool:
    """
    判断某个目录是否是一条合法 CARLA route。

    这里不依赖 route 文件夹名称，不要求必须以 Town 开头。
    只根据实际目录内容判断，因此未来即使路线名称改变，
    只要数据结构保持一致，仍然可以被正常发现。
    """
    if not path.is_dir():
        return False

    for name in REQUIRED_ROUTE_ENTRIES:
        if not (path / name).exists():
            return False

    if not (
        (path / "rgb_front").is_dir()
        or (path / "rgb").is_dir()
    ):
        return False

    return True


def _discover_routes_recursive(root: Path) -> List[Path]:
    """
    从 root 开始递归发现所有合法 route。

    使用 os.walk(topdown=True) 的原因：
    1. 可以遍历任意深度的数据集目录；
    2. 一旦发现当前目录已经是一条 route，就停止继续深入该 route，
       避免再扫描 rgb/、boxes/、measurements/ 等大量内部文件夹；
    3. 不依赖 "**/Town*" 这类文件名规则。

    默认 followlinks=False，避免符号链接形成递归环。
    """
    routes: List[Path] = []

    # 如果 input 本身恰好就是 route，也应直接识别。
    if is_route_dir(root):
        return [root.resolve()]

    for current_root, dirnames, _filenames in os.walk(
        str(root),
        topdown=True,
        followlinks=False,
    ):
        current = Path(current_root)

        if is_route_dir(current):
            routes.append(current.resolve())

            # 当前目录已经是 route。
            # 不再进入其 rgb/boxes/measurements 等内部目录，显著减少扫描开销。
            dirnames[:] = []
            continue

        # 这些目录通常体积大且不可能在内部再包含另一条 route。
        # 只有当它们出现在“非 route 父目录”中时才会被剪枝。
        # 这样可进一步降低大规模数据集递归扫描成本。
        prunable = {
            "rgb",
            "rgb_front",
            "instance_front",
            "boxes",
            "measurements",
            "cvaa_results",
            "cvaa_counterfactual_images",
            "cvaa_official_simlingo_inference",
            "cvaa_ad_fd",
        }
        dirnames[:] = [
            name
            for name in dirnames
            if name not in prunable
        ]

    return routes


def discover_routes(cfg: Dict[str, Any]) -> List[Path]:
    """
    根据 run.input / run.recursive 发现待处理 route。

    run.recursive=true：
        把 input 当作父目录，递归发现所有合法 route。

    run.recursive=false：
        把 input 本身直接作为唯一 route。

    最终仍会进行去重、排序以及 data.max_routes 截断。
    """
    input_path = Path(
        str(cfg["run"]["input"])
    ).expanduser().resolve()
    recursive = bool(cfg["run"]["recursive"])

    if not input_path.exists():
        raise FileNotFoundError(
            "run.input does not exist: %s" % input_path
        )

    if recursive:
        candidates = _discover_routes_recursive(input_path)
    else:
        candidates = [input_path]

    # 去重并再次检查 route 结构。
    valid: List[Path] = []
    invalid: List[Path] = []
    seen = set()

    for path in candidates:
        resolved = path.resolve()
        key = str(resolved)

        if key in seen:
            continue
        seen.add(key)

        if is_route_dir(resolved):
            valid.append(resolved)
        else:
            invalid.append(resolved)

    # 使用完整绝对路径排序，使递归发现顺序在不同运行中稳定。
    valid = sorted(
        valid,
        key=lambda p: str(p),
    )

    max_routes = int(
        cfg["data"].get("max_routes", 0) or 0
    )
    if max_routes > 0:
        valid = valid[:max_routes]

    if not valid:
        if recursive:
            msg = (
                "No valid CARLA route directories were recursively "
                "discovered under run.input=%s" % input_path
            )
        else:
            msg = (
                "run.input is not a valid CARLA route directory: %s"
                % input_path
            )

        if invalid:
            msg += " Checked examples: %s" % ", ".join(
                str(p) for p in invalid[:5]
            )

        raise RuntimeError(msg)

    return valid


def route_output_name(route_dir: Path) -> str:
    """
    默认以 route 文件夹名称作为输出名称。

    如果递归数据中不同父目录下存在同名 route，
    pipeline.py 会自动追加短路径哈希，避免结果目录冲突。
    """
    return route_dir.name


def resolved_paths(
    cfg: Dict[str, Any],
) -> Dict[str, Optional[Path]]:
    """把模型和输出相关路径统一解析为绝对路径。"""
    p = cfg["paths"]

    return {
        "output_root": _as_path(
            p["output_root"],
            "paths.output_root",
        ),
        "official_simlingo_root": _as_path(
            p["official_simlingo_root"],
            "paths.official_simlingo_root",
        ),
        "official_simlingo_checkpoint": _as_path(
            p["official_simlingo_checkpoint"],
            "paths.official_simlingo_checkpoint",
        ),
        "official_simlingo_config": _as_path(
            p.get("official_simlingo_config"),
            "paths.official_simlingo_config",
            allow_none=True,
        ),
        "lama_model": _as_path(
            p["lama_model"],
            "paths.lama_model",
        ),
        "temp_root": _as_path(
            p.get("temp_root"),
            "paths.temp_root",
            allow_none=True,
        ),
    }
