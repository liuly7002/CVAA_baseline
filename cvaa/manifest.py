from __future__ import annotations

import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_keyframe_manifest(
    data_root: Path,
    manifest_path: Path,
    max_routes: int = 0,
) -> Tuple[Dict[Path, List[str]], str]:
    """
    Parse the independent key-frame file.

    Expected line:
        relative/route/path/000123

    The route path is interpreted relative to run.input. No route discovery,
    recursive os.walk, or instance_front glob is performed here.
    """
    data_root = data_root.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()

    if not data_root.is_dir():
        raise FileNotFoundError("Dataset root does not exist: %s" % data_root)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "data.keyframe_file does not exist: %s" % manifest_path
        )

    grouped: "OrderedDict[Path, List[str]]" = OrderedDict()
    seen = set()

    with manifest_path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            value = raw.strip()
            if not value or value.startswith("#"):
                continue

            value = value.replace("\\", "/").rstrip("/")
            parts = [x for x in value.split("/") if x]
            if len(parts) < 2:
                raise ValueError(
                    "%s:%d invalid keyframe entry: %r"
                    % (manifest_path, line_no, value)
                )

            stem = parts[-1]
            route_rel = Path(*parts[:-1])
            route_dir = (data_root / route_rel).resolve()

            try:
                route_dir.relative_to(data_root)
            except Exception:
                raise ValueError(
                    "%s:%d escapes run.input: %r"
                    % (manifest_path, line_no, value)
                )

            key = (str(route_dir), stem)
            if key in seen:
                continue
            seen.add(key)
            grouped.setdefault(route_dir, []).append(stem)

    if not grouped:
        raise RuntimeError("No keyframes found in: %s" % manifest_path)

    result: "OrderedDict[Path, List[str]]" = OrderedDict()
    for route_dir, stems in grouped.items():
        if not route_dir.is_dir():
            raise FileNotFoundError(
                "Manifest route does not exist: %s" % route_dir
            )

        def sort_key(stem: str):
            try:
                return (0, int(stem))
            except Exception:
                return (1, stem)

        result[route_dir] = sorted(set(stems), key=sort_key)

    if int(max_routes or 0) > 0:
        result = OrderedDict(
            list(result.items())[: int(max_routes)]
        )

    return result, sha256_file(manifest_path)
