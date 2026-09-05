# CVAA Baseline for CARLA + Original SimLingo

A memory-bounded, batch-processing implementation of a **CVAA-style visual
counterfactual attribution baseline** for CARLA driving scenes.

The project turns the earlier development-stage scripts into one release-oriented
pipeline:

```text
CARLA actor / instance matching
        ↓
exact actor mask (in memory)
        ↓
LaMa object removal + FLUX seam refinement
        ↓
original SimLingo inference
        ↓
original / counterfactual trajectory comparison
        ↓
AD / FD
        ↓
per-frame actor ranking
```

The default configuration does **not** persist Stage-1/2/3 intermediate masks,
counterfactual images, or inference manifests. The complete workflow is run by a
single entry point and only compact final results plus optional debug figures are
kept.

---

## 1. Main design

### 1.1 Original SimLingo is used as the explained VLA model

The pipeline imports a clean checkout of:

- `https://github.com/RenzKa/simlingo`

and loads the released original SimLingo checkpoint with strict parameter
matching.

A source-tree guard rejects a modified `simlingo_liulei` tree when
`simlingo.allow_nonofficial_tree: false`.

### 1.2 CARLA ground truth replaces detector/segmenter discovery

For CARLA experiments, actor discovery and visual masks use simulator truth:

- actor metadata: `boxes/*.json.gz`
- front instance segmentation: `instance_front/*.png`
- camera metadata: `surround_camera_config.json`

The actor-to-instance mapping does **not** assume:

```text
CARLA actor id == image-side instance id
```

Instead, it combines projected 3D actor boxes, semantic compatibility, mask
containment, local dominance, centroid proximity, temporal consistency, and
Hungarian one-to-one assignment.

### 1.3 The visual counterfactual changes front RGB only

For actor \(A\):

```text
original:
    (RGB, speed, target point, prompt, calibration) -> SimLingo -> trajectory

counterfactual:
    (RGB without A, same speed, same target point, same prompt, same calibration)
    -> same SimLingo -> counterfactual trajectory
```

Pixels outside the intervention mask are forced to remain exactly equal to the
source image.

### 1.4 Formal ranking

The formal CVAA score uses `pred_route`:

```text
AD = mean_t || p_cf(t) - p_orig(t) ||_2
FD =          || p_cf(T) - p_orig(T) ||_2
```

Ranking inside each frame is:

1. `AD` descending
2. `FD` descending as tie-break

Ground-truth future waypoints are **not used for AD/FD ranking**. They are used
only for optional debug visualization.

---

## 2. Why this final version uses a unified bounded-cache pipeline

The development version wrote several large stages to disk:

```text
actor-instance JSON
→ exact masks
→ inpaint masks
→ counterfactual PNGs
→ inference JSON
→ AD/FD files
```

That is useful for debugging but wasteful for large-scale experiments.

The optimized final version processes one route in two isolated model phases:

```text
1. Match actors for the route in memory.
2. Take at most N actor interventions without splitting a frame.
3. Load LaMa/FLUX.
4. Generate only this chunk's counterfactual PNGs into a temporary directory.
5. Release LaMa/FLUX and free GPU memory.
6. Load original SimLingo.
7. Run original inference once per frame and counterfactual inference per actor.
8. Compute AD/FD immediately.
9. Save compact final scores/debug images.
10. Release SimLingo.
11. Delete the temporary chunk automatically.
12. Continue with the next chunk.
```

Therefore peak temporary disk usage is controlled by:

```yaml
runtime:
  max_counterfactuals_per_chunk: 16
```

This also prevents FLUX and SimLingo from occupying GPU memory at the same time,
which is important on approximately 16 GB GPUs.

---

## 3. Project structure

```text
cvaa_baseline_project/
├── config.yaml
├── run_pipeline.py
├── requirements_extra.txt
├── requirements_cvaa_fill.txt
├── README.md
├── RELEASE_NOTES.md
├── VALIDATION.md
├── THIRD_PARTY_NOTICES.md
├── .gitignore
├── cvaa/
│   ├── __init__.py
│   ├── config.py
│   ├── matching.py
│   ├── inpainting.py
│   ├── simlingo.py
│   ├── metrics.py
│   ├── pipeline.py
│   └── workers/
│       ├── inpaint_worker.py
│       └── simlingo_worker.py
└── tests/
    ├── test_metrics.py
    └── test_inpainting_core.py
```

There are no Stage-1/Stage-2/Stage-3 command-line tools in the final release.
Configuration is centralized in `config.yaml`.

---

## 4. Environment setup

### 4.1 Create the original SimLingo environment

The official SimLingo repository provides `environment.yaml` based on Python
3.8, PyTorch 2.2.0, Transformers 4.46.3, Accelerate 1.0.1, CARLA 0.9.15, etc.

```bash
git clone https://github.com/RenzKa/simlingo.git /path/to/RenzKa/simlingo
cd /path/to/RenzKa/simlingo

conda env create -f environment.yaml
conda activate simlingo
```

### 4.2 Install the extra CVAA dependency

From this project directory:

```bash
pip install -r requirements_extra.txt
```

The final project recommends:

```text
diffusers==0.32.2
```

because this release provides `FluxFillPipeline` and still supports Python 3.8.

The final project intentionally does **not** require the
`simple-lama-inpainting` Python package. A minimal TorchScript LaMa wrapper is
included in `cvaa/inpainting.py`, allowing `big-lama.pt` to run directly inside
the original SimLingo Python 3.8 environment.

### 4.3 Verify imports

```bash
python -c "import torch, transformers, accelerate, diffusers, cv2, scipy, hydra; print('environment OK')"
```

---

## 5. Required models

### 5.1 Original SimLingo

Download the official SimLingo model release, preserving the released directory
structure. The checkpoint is normally:

```text
simlingo/
└── checkpoints/
    └── epoch=013.ckpt/
        └── pytorch_model.pt
```

Example with Hugging Face CLI:

```bash
huggingface-cli download RenzKa/simlingo \
  --local-dir /path/to/simlingo_checkpoint
```

The corresponding released Hydra config is typically:

```text
/path/to/simlingo_checkpoint/simlingo/.hydra/config.yaml
```

### 5.2 FLUX.1-Fill-dev

Accept the model terms on Hugging Face first, then:

```bash
huggingface-cli login

huggingface-cli download black-forest-labs/FLUX.1-Fill-dev \
  --local-dir /path/to/FLUX.1-Fill-dev
```

### 5.3 LaMa

Download the TorchScript `big-lama.pt` model, for example from the
`simple-lama-inpainting` public release:

```bash
wget \
  https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt \
  -O /path/to/big-lama.pt
```

Model weights are not included in this repository.

---

## 6. Input route layout

Each CARLA route directory must contain:

```text
TownXX_.../
├── rgb/                     # or rgb_front/
│   ├── 0000.jpg
│   └── ...
├── instance_front/
│   ├── 0000.png
│   └── ...
├── boxes/
│   ├── 0000.json.gz
│   └── ...
├── measurements/
│   ├── 0000.json.gz
│   └── ...
└── surround_camera_config.json
```

The instance PNG must be the lossless CARLA front instance-segmentation image
collected by the corresponding data-collection pipeline.

---

## 7. Configuration

**All path selection, batch behavior, debug switches, model settings, and output
policy are edited in `config.yaml`. No folder/debug/model options are passed on
the command line.**

Every key in `config.yaml` already has an inline comment. The sections are:

| Section | Purpose |
|---|---|
| `run` | Input directory and recursive route discovery switch |
| `paths` | Output root, original SimLingo source/checkpoint, FLUX, LaMa, temporary cache |
| `data` | Route/frame range and candidate actor categories |
| `matching` | Actor-to-instance thresholds and temporal bonus |
| `mask` | Exact-mask filtering and adaptive dilation |
| `inpainting` | LaMa + FLUX settings, local crop, seam refinement, memory controls |
| `simlingo` | Original SimLingo device, seed, JPEG behavior, source-tree guard |
| `runtime` | Bounded chunk size, resume policy, progress logging |
| `debug` | Waypoint and inpainting visualizations |
| `output` | Which large artifacts, if any, are kept permanently |

### 7.1 Minimum fields to edit before the first run

At minimum set:

```yaml
run:
  input: "/your/data/simlingo"
  recursive: true

paths:
  output_root: "/your/cvaa_results"
  official_simlingo_root: "/your/RenzKa/simlingo"
  official_simlingo_checkpoint: "/your/pytorch_model.pt"
  flux_model: "/your/FLUX.1-Fill-dev"
  lama_model: "/your/big-lama.pt"
```

If the checkpoint directory does not preserve the original `.hydra/config.yaml`
layout, also set:

```yaml
paths:
  official_simlingo_config: "/path/to/config.yaml"
```


### 7.1.1 Recursive route discovery

The final release follows the same input style used by the language-grounded
waypoint project:

```yaml
run:
  input: "/path/to/data/simlingo"
  recursive: true
```

When `recursive: true`, `run.input` may point to a high-level dataset directory.
The pipeline recursively walks all subdirectories and recognizes a route by its
content rather than by a `Town*` name pattern. A valid route must contain:

```text
rgb/ or rgb_front/
instance_front/
boxes/
measurements/
surround_camera_config.json
```

Once a valid route directory is found, the recursive walker does not descend
into its internal image/measurement folders.

For single-route debugging:

```yaml
run:
  input: "/path/to/one/Town12_..."
  recursive: false
```

### 7.2 Low-disk recommended settings

```yaml
output:
  save_counterfactual_images: false
  save_masks: false
  save_trajectories: false

debug:
  enabled: true
  every_n_actors: 20
  save_waypoint_comparison: true
  save_inpainting_diagnostic: false

runtime:
  max_counterfactuals_per_chunk: 16
```

If temporary storage is especially limited, reduce
`max_counterfactuals_per_chunk` to `4` or `8`. This increases model reload
overhead but further reduces temporary disk usage.

### 7.3 Final paper inpainting settings

The current validated default is:

```yaml
inpainting:
  backend: "flux_fill"
  local_crop_enabled: true
  flux_refine_mode: "seam"
  sequential_cpu_offload: true
  vae_tiling: true
```

The intended process is:

```text
LaMa = actual object removal
FLUX = seam refinement only
```

This reduces the tendency of FLUX to hallucinate the removed vehicle back into
the scene.

---

## 8. Run

After editing `config.yaml`:

```bash
conda activate simlingo
cd /path/to/cvaa_baseline_project
python run_pipeline.py
```

That is the complete runtime command. There are no required command-line
parameters.

For a smoke test, edit the configuration instead of the command:

```yaml
data:
  max_routes: 1
  max_frames_per_route: 2
```

Then restore both values to `0` for full batch processing.

---

## 9. Final output

With the default low-disk policy:

```text
cvaa_results/
├── config_used.yaml
├── run_summary.json
├── all_actor_scores.jsonl
├── all_actor_scores.csv
├── all_frame_rankings.jsonl
└── <route_id>/
    ├── summary.json
    ├── actor_scores.jsonl
    ├── actor_scores.csv
    ├── frame_rankings.json
    ├── failures.jsonl                 # only if failures occur
    ├── skipped_interventions.jsonl    # only if filtering occurs
    └── debug/
        └── waypoints/
            ├── 0001_actor_3706.jpg
            └── ...
```

If enabled in `config.yaml`, the route directory may additionally contain:

```text
counterfactual_images/
masks/
debug/inpainting/
```

These are disabled by default to conserve disk space.

---

## 10. Debug visualization

When:

```yaml
debug:
  enabled: true
  save_waypoint_comparison: true
```

each selected actor can produce one compact image:

```text
LEFT
  original front RGB
  + original SimLingo pred_speed_wps
  + ground-truth future ego waypoints

RIGHT
  counterfactual front RGB
  + counterfactual SimLingo pred_speed_wps
  + the same ground-truth future ego waypoints
```

Color convention:

```text
RED   = predicted future waypoints
GREEN = ground-truth future waypoints
```

The ground truth is reconstructed from current/future `ego_matrix` values using
the same ego-frame transformation used by the original SimLingo dataset loader.

The ground truth is diagnostic only and does not enter formal CVAA AD/FD.

---

## 11. Resume behavior

For long batch runs:

```yaml
runtime:
  resume_completed_routes: true
  rebuild_incomplete_routes: true
```

A route whose `summary.json` contains:

```json
{"status": "complete"}
```

is skipped on rerun.

If a route was interrupted before completion, its compact partial output is
recreated rather than mixed with a later run.

Temporary chunk directories are deleted automatically.

---

## 12. Scientific invariants recorded by the implementation

The final implementation is designed around these invariants:

- only front RGB is counterfactually changed;
- outside intervention-mask pixels are bitwise unchanged before model
  preprocessing;
- original and counterfactual inference use the same original SimLingo network;
- original and counterfactual inference use the same checkpoint;
- speed, target point, next target point, prompt, camera calibration, and
  preprocessing are the same inside each pair;
- original inference is reused once per frame;
- formal actor ranking uses `pred_route` AD, with FD only as tie-break;
- `pred_speed_wps` AD/FD is supplementary only;
- future-waypoint ground truth is used only for visualization.

---

## 13. Recommended publication workflow

Before launching the full dataset:

```yaml
data:
  max_routes: 1
  max_frames_per_route: 2

debug:
  enabled: true
  every_n_actors: 1
```

Confirm that:

1. actor masks select the intended object;
2. counterfactual images remove the intended object;
3. debug trajectories project correctly;
4. `outside_mask_changed_pixels == 0`;
5. actor AD/FD ranking is generated.

Then set:

```yaml
data:
  max_routes: 0
  max_frames_per_route: 0
```

and run the complete batch.

For very large experiments, debug storage can be reduced without changing the
formal CVAA result:

```yaml
debug:
  every_n_actors: 20
```

or disabled completely:

```yaml
debug:
  enabled: false
```

---

## 14. Lightweight self-test

The project includes dependency-light metric and inpainting-core tests:

```bash
python -m unittest discover -s tests
```

This verifies AD/FD calculation, the `AD -> FD` ranking rule, adaptive-mask
behavior, and exact outside-mask pixel preservation. Full FLUX/LaMa + SimLingo
execution still requires the models and dataset described above.

---

## 15. Notes on reproducibility

The pipeline stores compact route and global summaries, actor matching scores,
mask metadata, AD/FD values, actor rankings, and optional model/debug metadata.

For a paper release, also archive:

- the exact `config.yaml` used for the reported experiment;
- original SimLingo Git commit/checkpoint;
- FLUX model version;
- `big-lama.pt` checksum;
- CARLA dataset generation version/commit.

See `THIRD_PARTY_NOTICES.md` for external dependencies and model sources.

> Repository owner: choose and add the project-level `LICENSE` appropriate for
> your public release before publishing. Third-party code/model licenses remain
> independent of that choice.



## 4.2 双环境运行架构（重要）

最终版本**不要把 FLUX / diffusers 安装到 `simlingo` 环境中**。

本项目使用两个已经隔离的 Conda 环境：

```text
simlingo
    └── Original SimLingo inference

cvaa_fill
    └── LaMa + FLUX.1-Fill-dev
```

用户仍然只需要：

```bash
conda activate simlingo
python run_pipeline.py
```

程序内部会根据 `config.yaml` 自动找到两个环境对应的 Python：

```yaml
environments:
  simlingo_conda_env: "simlingo"
  cvaa_fill_conda_env: "cvaa_fill"

  # 一般保持 null，程序自动从 Conda 环境解析。
  simlingo_python: null
  cvaa_fill_python: null

  validate_on_startup: true
```

如果某台机器无法自动解析 Conda 环境，也可以直接填写：

```yaml
environments:
  simlingo_python: "/.../envs/simlingo/bin/python"
  cvaa_fill_python: "/.../envs/cvaa_fill/bin/python"
```

### 实际执行顺序

```text
主进程
    ↓
当前 chunk 的 exact mask（临时文件）
    ↓
cvaa_fill worker
    ├── LaMa
    └── FLUX
    ↓
当前 chunk 的 counterfactual images（临时文件）
    ↓
cvaa_fill worker 退出
    ↓
GPU 显存释放
    ↓
simlingo worker
    └── Original SimLingo
    ↓
original / counterfactual prediction
    ↓
AD / FD / debug
    ↓
simlingo worker 退出
    ↓
删除当前 chunk 的临时 mask / counterfactual image / IPC JSON
```

因此：

- `simlingo` 不需要安装 `diffusers`；
- `cvaa_fill` 不需要加载 Original SimLingo；
- FLUX 和 SimLingo 不会同时驻留同一个 Python 进程；
- 每个 chunk 结束后临时反事实图仍会自动删除。

### cvaa_fill 环境

如果此前开发阶段的 `cvaa_fill` 已经能够运行 FLUX.1-Fill-dev，
建议继续直接使用，不要重新升级整套依赖。

可检查：

```bash
conda activate cvaa_fill
python -c "from diffusers import FluxFillPipeline; import diffusers; print(diffusers.__version__)"
```

应能正常导入 `FluxFillPipeline`。

如果只是缺少项目指定版本：

```bash
pip install -r requirements_cvaa_fill.txt
```

然后重新切回：

```bash
conda activate simlingo
python run_pipeline.py
```



## Route-level worker scheduling

The release uses two isolated Conda environments, but the heavy models are no
longer reloaded for every small chunk.

For each route:

```text
actor-instance matching
        ↓
one cvaa_fill worker
        ↓
load LaMa / FLUX once
        ↓
generate all temporary counterfactual images for this route
        ↓
cvaa_fill exits and releases GPU memory
        ↓
one simlingo worker
        ↓
load Original SimLingo / checkpoint once
        ↓
original inference once per frame
+
counterfactual inference per actor
        ↓
AD / FD / debug
        ↓
simlingo exits
        ↓
delete the route-level temporary cache
```

This keeps the scientific intervention unchanged while removing repeated FLUX
and SimLingo initialization. Temporary counterfactual images exist only for the
current route; they are deleted automatically when the route finishes unless
`output.save_counterfactual_images: true`.

If the system `/tmp` partition is small, set `paths.temp_root` to a local disk
with enough free space for one route.
