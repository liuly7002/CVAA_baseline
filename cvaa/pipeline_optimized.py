from __future__ import annotations

import csv
import hashlib
import json
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .config import resolved_environments, resolved_paths
from .manifest import load_keyframe_manifest
from .matching_optimized import match_route_actors_manifest
from .metrics import rank_actor_scores, scene_stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKER_DIR = PROJECT_ROOT / "cvaa" / "workers"


def _effective_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        k: v
        for k, v in cfg.items()
        if not str(k).startswith("_")
    }


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            value,
            f,
            ensure_ascii=False,
            indent=2,
        )


def _write_jsonl(
    path: Path,
    rows: List[Dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
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
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


def _route_path_hash(route_dir: Path) -> str:
    return hashlib.sha1(
        str(route_dir.resolve()).encode("utf-8")
    ).hexdigest()[:8]


def _build_route_output_map(
    routes: List[Path],
    output_root: Path,
) -> Dict[Path, Path]:
    counts = defaultdict(int)
    for route in routes:
        counts[route.name] += 1

    out = {}
    for route in routes:
        name = route.name
        if counts[name] > 1:
            name = "%s_%s" % (
                name,
                _route_path_hash(route),
            )
        out[route] = output_root / name
    return out


def _compute_run_signature(
    cfg: Dict[str, Any],
    manifest_sha256: str,
) -> str:
    effective = _effective_config(cfg)
    payload = {
        "config": effective,
        "keyframe_manifest_sha256": manifest_sha256,
        "pipeline": "persistent_dual_gpu_v1",
    }
    text = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(text).hexdigest()


def _preflight_environments(
    cfg: Dict[str, Any],
    envs: Dict[str, Path],
) -> None:
    if not bool(
        cfg["environments"].get(
            "validate_on_startup", True
        )
    ):
        return

    checks = [
        (
            "simlingo",
            envs["simlingo_python"],
            "import torch, transformers, accelerate",
        ),
        (
            "cvaa_fill",
            envs["cvaa_fill_python"],
            "import torch, diffusers; "
            "from diffusers import FluxFillPipeline",
        ),
    ]

    for name, python_path, code in checks:
        print(
            "[ENV] checking %s: %s"
            % (name, python_path)
        )
        proc = subprocess.run(
            [str(python_path), "-c", code],
            cwd=str(PROJECT_ROOT),
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "%s environment check failed." % name
            )


class PersistentWorker:
    READY = "@@CVAA_READY@@"
    DONE = "@@CVAA_DONE@@"

    def __init__(
        self,
        name: str,
        python_executable: Path,
        worker_script: Path,
        init_request: Path,
        physical_gpu: int,
    ) -> None:
        self.name = name
        self.done_queue: "queue.Queue[str]" = (
            queue.Queue()
        )
        self.ready = threading.Event()

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(
            int(physical_gpu)
        )
        old_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            str(PROJECT_ROOT)
            if not old_pythonpath
            else str(PROJECT_ROOT)
            + os.pathsep
            + old_pythonpath
        )

        self.proc = subprocess.Popen(
            [
                str(python_executable),
                str(worker_script),
                str(init_request),
            ],
            cwd=str(PROJECT_ROOT),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert self.proc.stdout is not None
        self.reader = threading.Thread(
            target=self._reader_loop,
            daemon=True,
        )
        self.reader.start()

        if not self.ready.wait(timeout=1800):
            raise RuntimeError(
                "%s worker did not become ready."
                % self.name
            )

    def _reader_loop(self) -> None:
        assert self.proc.stdout is not None
        for raw in self.proc.stdout:
            line = raw.rstrip("\n")
            if line == self.READY:
                self.ready.set()
                continue
            if line.startswith(self.DONE):
                self.done_queue.put(
                    line[len(self.DONE) :]
                )
                continue
            print(
                "[%s] %s" % (self.name, line),
                flush=True,
            )

    def process(
        self,
        job_id: str,
        request_path: Path,
        result_path: Path,
    ) -> Dict[str, Any]:
        if self.proc.poll() is not None:
            raise RuntimeError(
                "%s worker already exited with code %s"
                % (self.name, self.proc.returncode)
            )

        assert self.proc.stdin is not None
        command = {
            "op": "process",
            "job_id": str(job_id),
            "request_path": str(request_path),
            "result_path": str(result_path),
        }
        self.proc.stdin.write(
            json.dumps(command) + "\n"
        )
        self.proc.stdin.flush()

        while True:
            finished = self.done_queue.get()
            if finished == str(job_id):
                break

        if not result_path.exists():
            raise RuntimeError(
                "%s worker did not produce %s"
                % (self.name, result_path)
            )

        payload = json.loads(
            result_path.read_text(
                encoding="utf-8"
            )
        )
        if payload.get("status") == "fatal":
            raise RuntimeError(
                "%s worker fatal error: %s\n%s"
                % (
                    self.name,
                    payload.get("error"),
                    payload.get(
                        "traceback", ""
                    ),
                )
            )
        return payload

    def close(self) -> None:
        if self.proc.poll() is not None:
            return
        try:
            assert self.proc.stdin is not None
            self.proc.stdin.write(
                json.dumps({"op": "shutdown"})
                + "\n"
            )
            self.proc.stdin.flush()
            self.proc.wait(timeout=120)
        except Exception:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=10)
            except Exception:
                self.proc.kill()


def _prepare_route_output(
    route_output: Path,
    cfg: Dict[str, Any],
    run_signature: str,
) -> Tuple[bool, List[Dict[str, Any]]]:
    summary_path = route_output / "summary.json"

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
            summary.get("status") == "complete"
            and summary.get("run_signature")
            == run_signature
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
            f, fieldnames=fields
        )
        writer.writeheader()
        for score in scores:
            speed_diag = (
                score.get(
                    "speed_wps_diagnostic"
                )
                or {}
            )
            writer.writerow(
                {
                    "route_id": score.get(
                        "route_id"
                    ),
                    "frame": score.get("frame"),
                    "rank": score.get("rank"),
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
                    "speed_wps_AD": speed_diag.get(
                        "AD"
                    ),
                    "speed_wps_FD": speed_diag.get(
                        "FD"
                    ),
                    "exact_mask_pixels": score.get(
                        "exact_mask_pixels"
                    ),
                    "mask_pixels_used": score.get(
                        "mask_pixels_used"
                    ),
                    "counterfactual_image": score.get(
                        "counterfactual_image"
                    ),
                    "waypoint_debug_image": score.get(
                        "waypoint_debug_image"
                    ),
                }
            )


def _route_stats(
    actor_scores: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "AD": scene_stats(
            [float(x["AD"]) for x in actor_scores]
        ),
        "FD": scene_stats(
            [float(x["FD"]) for x in actor_scores]
        ),
    }


def _finalize_route(
    job: Dict[str, Any],
    sim_payload: Dict[str, Any],
    cfg: Dict[str, Any],
    run_signature: str,
    manifest_sha256: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    route_output = Path(job["route_output"])
    route_id = str(job["route_id"])
    actor_scores = list(
        sim_payload.get("scores", [])
    )

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
        route_output / "actor_scores.jsonl",
        actor_scores,
    )
    _write_actor_csv(
        route_output / "actor_scores.csv",
        actor_scores,
    )
    _write_jsonl(
        route_output / "frame_rankings.jsonl",
        frame_rankings,
    )

    failures = list(
        sim_payload.get("failures", [])
    )
    _write_jsonl(
        route_output / "failures.jsonl",
        failures,
    )

    summary = {
        "status": "complete",
        "route_id": route_id,
        "route_dir": str(job["route_dir"]),
        "keyframes_requested": int(
            job["keyframes_requested"]
        ),
        "matching": job["matching_summary"],
        "generated_counterfactuals": int(
            job.get(
                "generated_counterfactuals", 0
            )
        ),
        "evaluated_actors": len(actor_scores),
        "inference_failures": len(failures),
        "formal_ranking": {
            "metric": "pred_route",
            "primary": "AD descending",
            "tie_break": "FD descending",
            "uses_ground_truth": False,
        },
        "result_statistics": _route_stats(
            actor_scores
        ),
        "run_signature": run_signature,
        "keyframe_manifest_sha256": (
            manifest_sha256
        ),
        "execution": {
            "input_mode": "keyframe_manifest_direct",
            "persistent_workers": True,
            "dual_gpu_pipeline": True,
            "temporal_gap": (
                "raw_frame_number_difference"
            ),
        },
    }
    _json_dump(
        route_output / "summary.json",
        summary,
    )
    return summary, actor_scores


def run_pipeline(
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    start = time.time()
    paths = resolved_paths(cfg)
    envs = resolved_environments(cfg)
    _preflight_environments(cfg, envs)

    data_root = Path(
        str(cfg["run"]["input"])
    ).expanduser().resolve()
    manifest_path = Path(
        str(cfg["data"]["keyframe_file"])
    ).expanduser().resolve()

    manifest, manifest_sha256 = (
        load_keyframe_manifest(
            data_root=data_root,
            manifest_path=manifest_path,
            max_routes=int(
                cfg["data"].get(
                    "max_routes", 0
                )
                or 0
            ),
        )
    )

    run_signature = _compute_run_signature(
        cfg, manifest_sha256
    )

    output_root = paths["output_root"]
    assert output_root is not None
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    with (
        output_root / "config_used.yaml"
    ).open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            _effective_config(cfg),
            f,
            sort_keys=False,
            allow_unicode=True,
        )

    _json_dump(
        output_root
        / "benchmark_identity.json",
        {
            "keyframe_file": str(
                manifest_path
            ),
            "keyframe_manifest_sha256": (
                manifest_sha256
            ),
            "run_signature": run_signature,
            "routes": len(manifest),
            "keyframes": sum(
                len(x)
                for x in manifest.values()
            ),
        },
    )

    routes = list(manifest.keys())
    route_outputs = _build_route_output_map(
        routes, output_root
    )

    print("=" * 80)
    print("CVAA optimized key-frame pipeline")
    print("=" * 80)
    print(
        "manifest routes: %d" % len(routes)
    )
    print(
        "manifest keyframes: %d"
        % sum(
            len(x)
            for x in manifest.values()
        )
    )
    print(
        "manifest sha256: %s"
        % manifest_sha256
    )
    print(
        "GPU0 fill / GPU1 SimLingo: %s / %s"
        % (
            cfg["runtime"][
                "inpainting_gpu"
            ],
            cfg["runtime"][
                "simlingo_gpu"
            ],
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

    jobs: List[Dict[str, Any]] = []

    # --------------------------------------------------------------
    # CPU preparation. Only manifest frames are opened.
    # Matching and exact-mask eligibility share the same instance decode.
    # --------------------------------------------------------------
    for index, (
        route_dir,
        stems,
    ) in enumerate(manifest.items(), 1):
        route_output = route_outputs[
            route_dir
        ]
        route_id = route_output.name

        print(
            "[PREP %d/%d] %s (%d keyframes)"
            % (
                index,
                len(routes),
                route_dir,
                len(stems),
            ),
            flush=True,
        )

        try:
            skip, existing = (
                _prepare_route_output(
                    route_output,
                    cfg,
                    run_signature,
                )
            )
            if skip:
                all_scores.extend(existing)
                summary = json.loads(
                    (
                        route_output
                        / "summary.json"
                    ).read_text(
                        encoding="utf-8"
                    )
                )
                route_summaries.append(
                    summary
                )
                continue

            (
                valid_work,
                matching_summary,
                skipped,
            ) = match_route_actors_manifest(
                route_dir=route_dir,
                stems=stems,
                cfg=cfg,
            )
            _write_jsonl(
                route_output
                / "skipped_interventions.jsonl",
                skipped,
            )

            if not valid_work:
                empty_summary = {
                    "status": "complete",
                    "route_id": route_id,
                    "route_dir": str(
                        route_dir
                    ),
                    "keyframes_requested": len(
                        stems
                    ),
                    "matching": (
                        matching_summary
                    ),
                    "generated_counterfactuals": 0,
                    "evaluated_actors": 0,
                    "inference_failures": 0,
                    "formal_ranking": {
                        "metric": "pred_route",
                        "primary": (
                            "AD descending"
                        ),
                        "tie_break": (
                            "FD descending"
                        ),
                        "uses_ground_truth": False,
                    },
                    "result_statistics": (
                        _route_stats([])
                    ),
                    "run_signature": run_signature,
                    "keyframe_manifest_sha256": (
                        manifest_sha256
                    ),
                }
                _json_dump(
                    route_output
                    / "summary.json",
                    empty_summary,
                )
                _write_jsonl(
                    route_output
                    / "actor_scores.jsonl",
                    [],
                )
                _write_jsonl(
                    route_output
                    / "frame_rankings.jsonl",
                    [],
                )
                route_summaries.append(
                    empty_summary
                )
                continue

            temp_root = paths["temp_root"]
            if temp_root is not None:
                temp_root.mkdir(
                    parents=True,
                    exist_ok=True,
                )
            tmp_dir = Path(
                tempfile.mkdtemp(
                    prefix="cvaa_route_",
                    dir=(
                        str(temp_root)
                        if temp_root
                        is not None
                        else None
                    ),
                )
            )

            inpaint_items = []
            debug_counter = 0
            for item in valid_work:
                frame = str(item["frame"])
                actor_id = str(
                    item["actor_id"]
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
                        / "counterfactuals"
                        / frame
                        / (
                            "actor_%s.png"
                            % actor_id
                        )
                    )

                debug_counter += 1
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
                        debug_counter
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

                inpaint_items.append(
                    {
                        **item,
                        "counterfactual_image": str(
                            cf_path
                        ),
                        "diagnostic_path": (
                            str(diagnostic_path)
                            if diagnostic_path
                            is not None
                            else None
                        ),
                        "save_exact_mask_path": None,
                        "save_intervention_mask_path": None,
                    }
                )

            jobs.append(
                {
                    "job_id": (
                        "%06d_%s"
                        % (index, route_id)
                    ),
                    "route_dir": str(
                        route_dir
                    ),
                    "route_output": str(
                        route_output
                    ),
                    "route_id": route_id,
                    "keyframes_requested": (
                        len(stems)
                    ),
                    "matching_summary": (
                        matching_summary
                    ),
                    "tmp_dir": str(tmp_dir),
                    "inpaint_items": (
                        inpaint_items
                    ),
                }
            )

        except Exception as exc:
            failed_routes.append(
                {
                    "route_dir": str(
                        route_dir
                    ),
                    "error": repr(exc),
                }
            )
            print(
                "[PREP FAILED] %s: %s"
                % (route_dir, exc),
                flush=True,
            )

    if jobs:
        control_dir = Path(
            tempfile.mkdtemp(
                prefix="cvaa_workers_",
                dir=(
                    str(paths["temp_root"])
                    if paths["temp_root"]
                    is not None
                    else None
                ),
            )
        )

        fill_init = (
            control_dir / "fill_init.json"
        )
        sim_init = (
            control_dir / "sim_init.json"
        )

        _json_dump(
            fill_init,
            {
                "config": _effective_config(
                    cfg
                ),
                "lama_model_path": str(
                    paths["lama_model"]
                ),
                "flux_model": str(
                    cfg["paths"][
                        "flux_model"
                    ]
                ),
            },
        )
        _json_dump(
            sim_init,
            {
                "config": _effective_config(
                    cfg
                ),
                "official_simlingo_root": str(
                    paths[
                        "official_simlingo_root"
                    ]
                ),
                "official_simlingo_checkpoint": str(
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
            },
        )

        fill_worker = PersistentWorker(
            name="GPU-FILL",
            python_executable=envs[
                "cvaa_fill_python"
            ],
            worker_script=(
                WORKER_DIR
                / "inpaint_worker_persistent.py"
            ),
            init_request=fill_init,
            physical_gpu=int(
                cfg["runtime"][
                    "inpainting_gpu"
                ]
            ),
        )
        sim_worker = PersistentWorker(
            name="GPU-SIMLINGO",
            python_executable=envs[
                "simlingo_python"
            ],
            worker_script=(
                WORKER_DIR
                / "simlingo_worker_persistent.py"
            ),
            init_request=sim_init,
            physical_gpu=int(
                cfg["runtime"][
                    "simlingo_gpu"
                ]
            ),
        )

        buffer_routes = max(
            1,
            int(
                cfg["runtime"].get(
                    "pipeline_buffer_routes",
                    2,
                )
            ),
        )
        ready_for_sim: "queue.Queue[Optional[Dict[str, Any]]]" = (
            queue.Queue(
                maxsize=buffer_routes
            )
        )
        thread_errors: "queue.Queue[BaseException]" = (
            queue.Queue()
        )
        result_lock = threading.Lock()

        def fill_loop() -> None:
            try:
                for job in jobs:
                    tmp_dir = Path(
                        job["tmp_dir"]
                    )
                    request_path = (
                        tmp_dir
                        / "fill_request.json"
                    )
                    result_path = (
                        tmp_dir
                        / "fill_result.json"
                    )
                    _json_dump(
                        request_path,
                        {
                            "items": job[
                                "inpaint_items"
                            ]
                        },
                    )
                    payload = fill_worker.process(
                        job_id=str(
                            job["job_id"]
                        ),
                        request_path=request_path,
                        result_path=result_path,
                    )

                    request_index = {
                        (
                            str(x["frame"]),
                            str(
                                x["actor_id"]
                            ),
                        ): x
                        for x in job[
                            "inpaint_items"
                        ]
                    }

                    generated = []
                    fill_failures = []
                    for result in payload.get(
                        "results", []
                    ):
                        key = (
                            str(
                                result["frame"]
                            ),
                            str(
                                result[
                                    "actor_id"
                                ]
                            ),
                        )
                        original = (
                            request_index[key]
                        )
                        if (
                            result.get(
                                "status"
                            )
                            == "ok"
                        ):
                            meta = (
                                result.get(
                                    "meta"
                                )
                                or {}
                            )
                            generated.append(
                                {
                                    **original,
                                    **meta,
                                    "counterfactual_image": (
                                        original[
                                            "counterfactual_image"
                                        ]
                                    ),
                                }
                            )
                        else:
                            fill_failures.append(
                                result
                            )

                    job[
                        "generated_counterfactuals"
                    ] = len(generated)
                    job["generated_items"] = (
                        generated
                    )
                    job["fill_failures"] = (
                        fill_failures
                    )

                    ready_for_sim.put(job)

            except BaseException as exc:
                thread_errors.put(exc)
            finally:
                ready_for_sim.put(None)

        def sim_loop() -> None:
            try:
                while True:
                    job = ready_for_sim.get()
                    if job is None:
                        break

                    tmp_dir = Path(
                        job["tmp_dir"]
                    )
                    generated = job.get(
                        "generated_items", []
                    )

                    if generated:
                        request_path = (
                            tmp_dir
                            / "sim_request.json"
                        )
                        result_path = (
                            tmp_dir
                            / "sim_result.json"
                        )
                        _json_dump(
                            request_path,
                            {
                                "route_dir": job[
                                    "route_dir"
                                ],
                                "route_output": job[
                                    "route_output"
                                ],
                                "route_id": job[
                                    "route_id"
                                ],
                                "items": generated,
                            },
                        )
                        sim_payload = (
                            sim_worker.process(
                                job_id=str(
                                    job[
                                        "job_id"
                                    ]
                                ),
                                request_path=(
                                    request_path
                                ),
                                result_path=(
                                    result_path
                                ),
                            )
                        )
                    else:
                        sim_payload = {
                            "status": "complete",
                            "scores": [],
                            "failures": [],
                        }

                    fill_failures = list(
                        job.get(
                            "fill_failures", []
                        )
                    )
                    if fill_failures:
                        existing = list(
                            sim_payload.get(
                                "failures", []
                            )
                        )
                        for failure in (
                            fill_failures
                        ):
                            existing.append(
                                {
                                    "stage": (
                                        "inpainting"
                                    ),
                                    "frame": failure.get(
                                        "frame"
                                    ),
                                    "actor_id": failure.get(
                                        "actor_id"
                                    ),
                                    "error": (
                                        failure.get(
                                            "error"
                                        )
                                        or failure.get(
                                            "reason"
                                        )
                                    ),
                                    "traceback": (
                                        failure.get(
                                            "traceback"
                                        )
                                    ),
                                }
                            )
                        sim_payload[
                            "failures"
                        ] = existing

                    summary, scores = (
                        _finalize_route(
                            job=job,
                            sim_payload=sim_payload,
                            cfg=cfg,
                            run_signature=(
                                run_signature
                            ),
                            manifest_sha256=(
                                manifest_sha256
                            ),
                        )
                    )

                    with result_lock:
                        route_summaries.append(
                            summary
                        )
                        all_scores.extend(
                            scores
                        )

                    shutil.rmtree(
                        tmp_dir,
                        ignore_errors=True,
                    )

            except BaseException as exc:
                thread_errors.put(exc)

        fill_thread = threading.Thread(
            target=fill_loop,
            name="cvaa-fill-producer",
        )
        sim_thread = threading.Thread(
            target=sim_loop,
            name="cvaa-sim-consumer",
        )

        try:
            fill_thread.start()
            sim_thread.start()
            fill_thread.join()
            sim_thread.join()

            if not thread_errors.empty():
                raise thread_errors.get()

        finally:
            fill_worker.close()
            sim_worker.close()
            shutil.rmtree(
                control_dir,
                ignore_errors=True,
            )
            for job in jobs:
                shutil.rmtree(
                    Path(job["tmp_dir"]),
                    ignore_errors=True,
                )

    global_rankings = rank_actor_scores(
        all_scores
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
    _write_actor_csv(
        output_root
        / "all_actor_scores.csv",
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
        "routes_manifest": len(routes),
        "keyframes_manifest": sum(
            len(x)
            for x in manifest.values()
        ),
        "routes_complete": len(
            route_summaries
        ),
        "routes_failed": len(
            failed_routes
        ),
        "evaluated_actor_interventions": (
            len(all_scores)
        ),
        "keyframe_manifest_sha256": (
            manifest_sha256
        ),
        "run_signature": run_signature,
        "failed_routes": failed_routes,
        "execution": {
            "persistent_workers": True,
            "dual_gpu_pipeline": True,
            "inpainting_gpu": int(
                cfg["runtime"][
                    "inpainting_gpu"
                ]
            ),
            "simlingo_gpu": int(
                cfg["runtime"][
                    "simlingo_gpu"
                ]
            ),
        },
    }
    _json_dump(
        output_root / "run_summary.json",
        run_summary,
    )

    print("=" * 80)
    print("CVAA optimized run complete")
    print("=" * 80)
    print(
        "routes complete: %d"
        % run_summary[
            "routes_complete"
        ]
    )
    print(
        "routes failed:   %d"
        % run_summary[
            "routes_failed"
        ]
    )
    print(
        "actor scores:    %d"
        % len(all_scores)
    )
    print(
        "elapsed:         %.1fs"
        % run_summary[
            "elapsed_seconds"
        ]
    )
    print("=" * 80)

    return run_summary
