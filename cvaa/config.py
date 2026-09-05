from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml


REQUIRED_ROUTE_ENTRIES = (
    "instance_front",
    "boxes",
    "measurements",
    "surround_camera_config.json",
)


def load_config(path: Path) -> Dict[str, Any]:
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
    cur: Any = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            raise KeyError("Missing required config key: %s" % dotted)
        cur = cur[key]
    return cur


def _as_path(value: Any, name: str, allow_none: bool = False) -> Optional[Path]:
    if value is None:
        if allow_none:
            return None
        raise ValueError("%s may not be null" % name)
    return Path(str(value)).expanduser().resolve()


def validate_config(cfg: Dict[str, Any]) -> None:
    for section in (
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

    explicit_routes = _require(cfg, "paths.explicit_routes")
    if not isinstance(explicit_routes, list):
        raise TypeError("paths.explicit_routes must be a list")

    if not explicit_routes:
        routes_root = _as_path(
            _require(cfg, "paths.routes_root"),
            "paths.routes_root",
        )
        if routes_root is not None and not routes_root.exists():
            raise FileNotFoundError(
                "paths.routes_root does not exist: %s" % routes_root
            )
        if not str(_require(cfg, "paths.route_glob")).strip():
            raise ValueError("paths.route_glob may not be empty")
    else:
        for value in explicit_routes:
            route_path = Path(str(value)).expanduser().resolve()
            if not route_path.exists():
                raise FileNotFoundError(
                    "Explicit route does not exist: %s" % route_path
                )

    _as_path(_require(cfg, "paths.output_root"), "paths.output_root")
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
            raise FileNotFoundError("%s does not exist: %s" % (name, path))

    if explicit_sim_cfg is not None and not explicit_sim_cfg.exists():
        raise FileNotFoundError(
            "paths.official_simlingo_config does not exist: %s"
            % explicit_sim_cfg
        )

    flux_value = str(_require(cfg, "paths.flux_model"))
    flux_path = Path(flux_value).expanduser()
    if flux_path.is_absolute() and not flux_path.exists():
        raise FileNotFoundError(
            "paths.flux_model local directory does not exist: %s"
            % flux_path
        )

    backend = str(_require(cfg, "inpainting.backend"))
    if backend not in ("flux_fill", "lama_only", "opencv"):
        raise ValueError("Unsupported inpainting.backend=%r" % backend)

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
        raise ValueError("mask.adaptive_dilate_ratio must be >= 0")


def is_route_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    for name in REQUIRED_ROUTE_ENTRIES:
        if not (path / name).exists():
            return False
    if not ((path / "rgb_front").is_dir() or (path / "rgb").is_dir()):
        return False
    return True


def discover_routes(cfg: Dict[str, Any]) -> List[Path]:
    paths_cfg = cfg["paths"]
    explicit = paths_cfg.get("explicit_routes") or []

    candidates: List[Path] = []
    if explicit:
        for value in explicit:
            candidates.append(Path(str(value)).expanduser().resolve())
    else:
        root = Path(str(paths_cfg["routes_root"])).expanduser().resolve()
        pattern = str(paths_cfg["route_glob"])
        if not root.exists():
            raise FileNotFoundError("paths.routes_root does not exist: %s" % root)
        candidates.extend(sorted(root.glob(pattern)))

    valid: List[Path] = []
    invalid: List[Path] = []
    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if str(resolved) in seen:
            continue
        seen.add(str(resolved))
        if is_route_dir(resolved):
            valid.append(resolved)
        else:
            invalid.append(resolved)

    max_routes = int(cfg["data"].get("max_routes", 0) or 0)
    if max_routes > 0:
        valid = valid[:max_routes]

    if not valid:
        msg = "No valid CARLA route directories were discovered."
        if invalid:
            msg += " Checked examples: %s" % ", ".join(
                str(p) for p in invalid[:5]
            )
        raise RuntimeError(msg)

    return valid


def route_output_name(route_dir: Path) -> str:
    """
    Route timestamped names are normally unique. If users have duplicated route
    names under different parents, a short path hash is appended by pipeline.py.
    """
    return route_dir.name


def resolved_paths(cfg: Dict[str, Any]) -> Dict[str, Optional[Path]]:
    p = cfg["paths"]
    return {
        "output_root": _as_path(p["output_root"], "paths.output_root"),
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
        "lama_model": _as_path(p["lama_model"], "paths.lama_model"),
        "temp_root": _as_path(
            p.get("temp_root"),
            "paths.temp_root",
            allow_none=True,
        ),
    }
