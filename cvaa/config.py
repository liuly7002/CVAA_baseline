from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
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
        "environments",
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
    # 双 Python 环境
    # ------------------------------------------------------------------
    env_cfg = cfg["environments"]

    for key in (
        "simlingo_conda_env",
        "cvaa_fill_conda_env",
    ):
        value = env_cfg.get(key)
        if value is None or not str(value).strip():
            raise ValueError(
                "environments.%s may not be empty" % key
            )

    for key in (
        "simlingo_python",
        "cvaa_fill_python",
    ):
        value = env_cfg.get(key)
        if value is not None:
            python_path = _as_path(
                value,
                "environments.%s" % key,
            )
            if python_path is None or not python_path.is_file():
                raise FileNotFoundError(
                    "environments.%s does not exist: %s"
                    % (key, python_path)
                )
            if not os.access(str(python_path), os.X_OK):
                raise PermissionError(
                    "environments.%s is not executable: %s"
                    % (key, python_path)
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



def _conda_env_prefixes() -> Dict[str, Path]:
    """
    读取当前 Conda 可见的环境。

    返回：
        {
            "simlingo": Path("/.../envs/simlingo"),
            "cvaa_fill": Path("/.../envs/cvaa_fill"),
            ...
        }

    优先使用 CONDA_EXE；如果当前 shell 没有该变量，再从 PATH 中查找 conda。
    """
    conda_exe = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if not conda_exe:
        raise RuntimeError(
            "无法找到 conda。请先初始化 Conda shell，"
            "或者在 config.yaml 中直接填写 environments.*_python。"
        )

    proc = subprocess.run(
        [str(conda_exe), "env", "list", "--json"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "执行 'conda env list --json' 失败：%s"
            % proc.stderr.strip()
        )

    try:
        payload = json.loads(proc.stdout)
    except Exception as exc:
        raise RuntimeError(
            "无法解析 Conda 环境列表：%r" % exc
        )

    result: Dict[str, Path] = {}
    for raw_prefix in payload.get("envs", []):
        prefix = Path(str(raw_prefix)).expanduser().resolve()
        name = prefix.name
        result[name] = prefix

    # base 环境的 prefix 名称通常不是 "base"，额外根据 CONDA_PREFIX 补充。
    current_env = os.environ.get("CONDA_DEFAULT_ENV")
    current_prefix = os.environ.get("CONDA_PREFIX")
    if current_env and current_prefix:
        result[str(current_env)] = (
            Path(current_prefix).expanduser().resolve()
        )

    return result


def _python_from_conda_env(env_name: str) -> Path:
    """根据 Conda 环境名称找到对应的 python 可执行文件。"""
    env_name = str(env_name).strip()

    # 当前进程本身就在目标环境时，直接使用 sys.executable，
    # 避免不必要的 Conda 查询。
    if os.environ.get("CONDA_DEFAULT_ENV") == env_name:
        current = Path(sys.executable).expanduser().resolve()
        if current.is_file():
            return current

    prefixes = _conda_env_prefixes()
    if env_name not in prefixes:
        raise RuntimeError(
            "Conda 环境 %r 不存在。当前可见环境：%s"
            % (
                env_name,
                ", ".join(sorted(prefixes.keys())),
            )
        )

    prefix = prefixes[env_name]

    # Linux / macOS
    candidate = prefix / "bin" / "python"
    if candidate.is_file():
        return candidate.resolve()

    # Windows 兼容
    candidate = prefix / "python.exe"
    if candidate.is_file():
        return candidate.resolve()

    raise FileNotFoundError(
        "在 Conda 环境 %r 中找不到 python：%s"
        % (env_name, prefix)
    )


def _resolve_worker_python(
    explicit_python: Any,
    conda_env_name: Any,
    field_name: str,
) -> Path:
    """
    worker Python 的解析顺序：

    1. 如果 config.yaml 显式填写了 python 完整路径，直接使用；
    2. 否则根据 Conda 环境名称自动解析。
    """
    if explicit_python is not None:
        path = Path(str(explicit_python)).expanduser().resolve()
    else:
        path = _python_from_conda_env(str(conda_env_name))

    if not path.is_file():
        raise FileNotFoundError(
            "%s does not exist: %s" % (field_name, path)
        )
    if not os.access(str(path), os.X_OK):
        raise PermissionError(
            "%s is not executable: %s" % (field_name, path)
        )
    return path


def resolved_environments(
    cfg: Dict[str, Any],
) -> Dict[str, Path]:
    """
    解析两个隔离 worker 的 Python。

    最终运行架构：
        parent process
            ├── cvaa_fill python -> LaMa + FLUX
            └── simlingo python  -> Original SimLingo
    """
    env_cfg = cfg["environments"]

    return {
        "simlingo_python": _resolve_worker_python(
            env_cfg.get("simlingo_python"),
            env_cfg.get("simlingo_conda_env"),
            "environments.simlingo_python",
        ),
        "cvaa_fill_python": _resolve_worker_python(
            env_cfg.get("cvaa_fill_python"),
            env_cfg.get("cvaa_fill_conda_env"),
            "environments.cvaa_fill_python",
        ),
    }


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
