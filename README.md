# CVAA Baseline for CARLA + Original SimLingo

> 中文说明版 README  
> 本文件与英文版 `README.md` 对应，用于说明项目的设计目的、环境搭建、配置方式、批量运行流程、输出内容以及复现实验时需要注意的事项。  
> 代码中的类名、配置字段、文件名和模型名保持英文不变，避免与实际项目实现产生歧义。

这是一个面向 CARLA 驾驶场景的、**显存/磁盘占用受控的批量 CVAA 风格视觉反事实归因基线项目**。

该项目将此前开发阶段中相互独立的 Stage 1 / Stage 2 / Stage 3 / Stage 4 脚本整理为一套统一的正式处理流程：

```text
CARLA actor / instance 匹配
        ↓
精确 actor mask（以内存形式保留）
        ↓
LaMa 删除目标 + FLUX 边界修复
        ↓
原始官方 SimLingo 推理
        ↓
原始轨迹 / 反事实轨迹比较
        ↓
AD / FD 计算
        ↓
每帧 actor 排序
```

默认配置下，项目**不会永久保存 Stage 1 / Stage 2 / Stage 3 的大规模中间文件**，例如：

- actor-instance 中间 JSON；
- mask PNG；
- 反事实 PNG；
- 中间 inference manifest。

完整流程由一个入口统一运行，最终仅保留：

- actor 的 AD / FD；
- 每帧 actor 排名；
- 路线级/全局级摘要；
- 可选 debug 图。

这样可以显著降低大规模实验时的本地磁盘占用。

---

# 1. 项目设计原则

## 1.1 被解释模型使用原始官方 SimLingo

本项目中的 CVAA baseline 使用一个**干净的官方 SimLingo 仓库**：

```text
https://github.com/RenzKa/simlingo
```

并加载官方发布的原始 SimLingo checkpoint。

这里特别强调：

> 本项目中的 CVAA baseline 不使用 `simlingo_liulei` 中经过论文修改后的网络，而是使用原始官方 SimLingo。

这样可以将该基线明确表示为：

```text
CVAA-style visual counterfactual attribution
+
Original SimLingo
```

配置中：

```yaml
simlingo:
  allow_nonofficial_tree: false
```

时，程序会检查 `official_simlingo_root`。

如果误传入修改后的 `simlingo_liulei`，并检测到例如：

```text
future_interaction_grid
camera_attention_target
counterfactual_waypoints
FutureInteractionDecoder
PreLanguageInteractionReasoner
```

等修改内容，程序会拒绝继续运行。

该检查用于防止最终论文实验中误把修改后的网络当作官方 SimLingo baseline。

---

## 1.2 CARLA 仿真真值替代检测器和分割器

原始 CVAA 需要从前视图中：

```text
目标检测
→ 目标分割
→ 对象删除
```

而在 CARLA 中，我们已经拥有准确的仿真真值，因此本项目使用：

```text
boxes/*.json.gz
```

作为 actor 几何与类别信息，并使用：

```text
instance_front/*.png
```

作为前视 instance segmentation。

相机参数来自：

```text
surround_camera_config.json
```

需要注意：

```text
CARLA actor id != 图像中的 instance id
```

程序**不会直接假定这两个 ID 相等**。

actor 与 instance 的匹配会综合考虑：

- 3D actor bbox 投影；
- semantic 类别兼容性；
- instance mask 在投影区域中的覆盖率；
- 投影框内部局部主导程度；
- mask / bbox 中心距离；
- 时间连续性；
- Hungarian 一对一匹配。

因此最终获得的是：

```text
CARLA actor
        ↕
front instance segmentation object
```

之间的可靠对应关系。

---

## 1.3 反事实干预只允许改变 front RGB

对于 actor \(A\)，原始输入为：

```text
RGB
speed
target point
next target point
prompt
camera calibration
```

原始推理：

```text
(original RGB, non-visual context)
        ↓
Original SimLingo
        ↓
trajectory
```

删除 actor \(A\) 后：

```text
(counterfactual RGB, same non-visual context)
        ↓
同一个 Original SimLingo
        ↓
counterfactual trajectory
```

因此：

\[
(I,z)\rightarrow f_{\text{SimLingo}}\rightarrow \tau
\]

\[
(I^{-A},z)\rightarrow f_{\text{SimLingo}}\rightarrow \tau^{-A}
\]

其中：

- \(I\)：原始前视图像；
- \(I^{-A}\)：删除 actor \(A\) 后的反事实图像；
- \(z\)：speed、target point、prompt、camera calibration 等非视觉输入；
- \(\tau\)：原始模型轨迹；
- \(\tau^{-A}\)：反事实轨迹。

整个实验中：

```text
z 完全保持不变
```

只有：

```text
front RGB
```

发生改变。

此外，最终反事实图会进行严格像素约束：

> intervention mask 之外的像素必须与原图完全一致。

也就是说，反事实干预被限制在目标 actor 对应区域内。

---

## 1.4 正式 CVAA 排名指标

正式 actor 排名使用：

```text
pred_route
```

计算 AD 和 FD。

对于原始平均轨迹：

\[
\bar p_t
\]

以及删除 actor 后的平均轨迹：

\[
\bar p_t^{cf}
\]

定义：

\[
AD=
\frac{1}{T}
\sum_{t=1}^{T}
\left\|
\bar p_t^{cf}-\bar p_t
\right\|_2
\]

\[
FD=
\left\|
\bar p_T^{cf}-\bar p_T
\right\|_2
\]

每一帧内部的 actor 排名规则为：

```text
第一排序指标：AD，从大到小
第二排序指标：FD，从大到小，仅用于 AD 相同时的 tie-break
```

因此：

```text
rank 1 = 当前场景中对 SimLingo 驾驶轨迹影响最大的 actor
```

### 关于 Ground Truth

GT future waypoints：

```text
不参与 AD / FD
不参与正式 actor 排名
```

GT 只用于 debug 图像中，帮助直观比较：

```text
预测轨迹 vs 真实轨迹
```

---

# 2. 为什么最终版本采用统一的 bounded-cache 流程

开发阶段为了便于调试，每一个 Stage 都会保存大量中间文件：

```text
actor-instance JSON
→ exact mask
→ inpaint mask
→ counterfactual PNG
→ inference JSON
→ AD / FD
```

这种方式适合开发，但不适合大规模论文实验。

例如，一条路线可能包含：

```text
数百帧
×
每帧多个 actor
```

如果每个 actor 都永久保存：

```text
mask
counterfactual image
debug image
JSON
```

那么磁盘占用会快速增长。

因此最终版本采用**分块处理 + 临时缓存自动释放**的设计。

实际执行流程为：

```text
1. 对当前 route 在内存中完成 actor-instance 匹配。

2. 按 chunk 选择一批 actor intervention。
   一个 frame 不会被拆到两个 chunk 中。

3. 加载 LaMa / FLUX。

4. 仅生成当前 chunk 所需的 counterfactual image，
   保存到临时目录。

5. 当前 chunk 的反事实图生成完成后，
   释放 LaMa / FLUX，并清理 GPU 内存。

6. 加载原始官方 SimLingo。

7. 每个 frame 的 original image 只推理一次。

8. 当前 frame 下每个 actor 的 counterfactual image
   分别进行 SimLingo 推理。

9. 立即计算：
       AD
       FD
       actor rank

10. 只保存紧凑的最终结果和可选 debug。

11. 释放 SimLingo。

12. 删除当前 chunk 的临时反事实图。

13. 继续处理下一 chunk。
```

因此，同一时间永久占用磁盘的反事实图数量最多约为：

```yaml
runtime:
  max_counterfactuals_per_chunk: 16
```

如果本地临时磁盘空间更紧张，可以设置：

```yaml
runtime:
  max_counterfactuals_per_chunk: 4
```

或者：

```yaml
runtime:
  max_counterfactuals_per_chunk: 8
```

代价是：

```text
模型重新加载次数增加
```

但磁盘占用会进一步下降。

该设计还有一个重要优点：

> FLUX / LaMa 和 SimLingo 不需要同时驻留 GPU。

对于约 16 GB 显存的 GPU，这一点非常重要。

---

# 3. 项目目录结构

最终项目结构如下：

```text
cvaa_baseline_project/
├── config.yaml
├── run_pipeline.py
├── requirements_extra.txt
├── README.md
├── README_zh.md
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
│   └── pipeline.py
└── tests/
    ├── test_metrics.py
    └── test_inpainting_core.py
```

各文件主要作用如下。

### `config.yaml`

整个项目唯一需要经常修改的配置文件。

包括：

```text
数据路径
模型路径
actor 类别
mask 参数
LaMa / FLUX 参数
SimLingo 参数
chunk 大小
resume
debug
输出策略
```

最终版本不再依赖大量命令行参数。

---

### `run_pipeline.py`

项目唯一正式运行入口：

```bash
python run_pipeline.py
```

---

### `cvaa/matching.py`

负责：

```text
CARLA actor
↔
front instance segmentation
```

匹配。

---

### `cvaa/inpainting.py`

负责：

```text
actor mask
→ LaMa object removal
→ FLUX seam refinement
→ counterfactual image
```

---

### `cvaa/simlingo.py`

负责：

```text
加载 Original SimLingo
构建模型输入
original inference
counterfactual inference
GT waypoint debug projection
```

---

### `cvaa/metrics.py`

负责：

```text
AD
FD
actor ranking
```

---

### `cvaa/pipeline.py`

负责整个 batch workflow：

```text
route discovery
→ matching
→ mask
→ inpainting
→ inference
→ AD / FD
→ ranking
→ output
```

---

# 4. 环境搭建

## 4.1 创建原始 SimLingo 环境

首先克隆官方 SimLingo：

```bash
git clone https://github.com/RenzKa/simlingo.git /path/to/RenzKa/simlingo
```

进入官方目录：

```bash
cd /path/to/RenzKa/simlingo
```

使用官方提供的：

```text
environment.yaml
```

创建环境：

```bash
conda env create -f environment.yaml
```

激活：

```bash
conda activate simlingo
```

官方环境主要包括：

```text
Python 3.8
PyTorch 2.2.0
Transformers 4.46.3
Accelerate 1.0.1
CARLA 0.9.15
Hydra
OpenCV
SciPy
```

等。

> 建议最终 CVAA 实验始终在这个官方 SimLingo 环境中运行，而不要直接在后来修改过的训练环境中进行。

---

## 4.2 安装 CVAA 额外依赖

进入当前 CVAA 项目：

```bash
cd /path/to/cvaa_baseline_project
```

执行：

```bash
pip install -r requirements_extra.txt
```

当前项目推荐：

```text
diffusers==0.32.2
```

因为该版本：

```text
支持 FluxFillPipeline
并且仍支持 Python 3.8
```

---

## 4.3 关于 LaMa

最终项目**不要求安装**：

```text
simple-lama-inpainting
```

Python 包。

原因是该包对新版本 Python 的依赖可能与官方 SimLingo Python 3.8 环境冲突。

项目已经在：

```text
cvaa/inpainting.py
```

中包含一个精简的 LaMa TorchScript 推理封装。

因此只需要准备：

```text
big-lama.pt
```

即可。

---

## 4.4 环境检查

安装完成后可以执行：

```bash
python -c "import torch, transformers, accelerate, diffusers, cv2, scipy, hydra; print('environment OK')"
```

如果输出：

```text
environment OK
```

说明主要依赖可以正常导入。

---

# 5. 必需模型

## 5.1 Original SimLingo

下载官方 SimLingo 模型。

推荐保留官方目录结构，例如：

```text
simlingo/
├── .hydra/
│   └── config.yaml
└── checkpoints/
    └── epoch=013.ckpt/
        └── pytorch_model.pt
```

可以使用：

```bash
huggingface-cli download RenzKa/simlingo \
  --local-dir /path/to/simlingo_checkpoint
```

最终 checkpoint 通常为：

```text
/path/to/simlingo_checkpoint/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt
```

对应 Hydra 配置通常为：

```text
/path/to/simlingo_checkpoint/simlingo/.hydra/config.yaml
```

如果 checkpoint 保存目录仍然保持官方层级结构，项目可以自动推断 `.hydra/config.yaml`。

否则需要在 `config.yaml` 中手动填写：

```yaml
paths:
  official_simlingo_config: "/path/to/config.yaml"
```

---

## 5.2 FLUX.1-Fill-dev

首先在 Hugging Face 上接受：

```text
black-forest-labs/FLUX.1-Fill-dev
```

对应模型协议。

然后登录：

```bash
huggingface-cli login
```

下载：

```bash
huggingface-cli download black-forest-labs/FLUX.1-Fill-dev \
  --local-dir /path/to/FLUX.1-Fill-dev
```

在配置中填写：

```yaml
paths:
  flux_model: "/path/to/FLUX.1-Fill-dev"
```

---

## 5.3 LaMa

准备：

```text
big-lama.pt
```

例如：

```bash
wget \
  https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt \
  -O /path/to/big-lama.pt
```

然后配置：

```yaml
paths:
  lama_model: "/path/to/big-lama.pt"
```

模型权重本身不会包含在本项目 Git 仓库中。

---

# 6. 输入数据结构

每一个 CARLA route 目录应包含：

```text
TownXX_.../
├── rgb/                     # 或 rgb_front/
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

各目录含义如下。

### `rgb/`

原始前视 RGB 图像。

---

### `instance_front/`

CARLA front instance segmentation。

必须是：

```text
lossless PNG
```

不能转成 JPEG，否则 instance id 会被破坏。

---

### `boxes/`

CARLA actor 仿真真值，包括：

```text
class
position
extent
yaw
actor id
transform / matrix
```

等信息。

---

### `measurements/`

自车状态与导航信息，例如：

```text
speed
target_point
target_point_next
route
ego_matrix
```

其中 future `ego_matrix` 也用于 debug GT future waypoint 的构造。

---

### `surround_camera_config.json`

保存相机外参与分辨率/FOV 等信息，用于 actor bbox 投影和 instance 匹配。

---

# 7. 配置文件 `config.yaml`

**正式运行时，所有路径、批量行为、debug、模型参数以及输出策略均通过 `config.yaml` 设置。**

不需要再通过：

```bash
--route-dir
--checkpoint
--debug
--limit
```

等命令行参数控制。

配置文件主要分为以下部分：

| 配置部分 | 功能 |
|---|---|
| `run` | 数据输入路径以及是否递归查找 route |
| `paths` | 结果目录、官方 SimLingo、FLUX、LaMa、临时缓存路径 |
| `data` | route/frame 范围、候选 actor 类别 |
| `matching` | actor-instance 匹配阈值和时间连续性参数 |
| `mask` | exact mask 过滤以及 dilation |
| `inpainting` | LaMa + FLUX、crop、seam refinement、显存优化 |
| `simlingo` | Original SimLingo 推理设备、随机种子、JPEG 策略、官方源码检查 |
| `runtime` | chunk 大小、resume、日志频率 |
| `debug` | waypoint / inpainting 可视化 |
| `output` | 是否永久保留 mask、counterfactual image、trajectory 等大文件 |

每一个配置字段在 `config.yaml` 中都已经附有简短注释。

---

## 7.1 第一次运行前最少需要修改的内容

至少修改：

```yaml
run:
  # 可以填写 data/simlingo 等高层父目录。
  input: "/your/data/simlingo"

  # true：递归寻找 input 下所有合法 route。
  # false：直接把 input 当作一条 route。
  recursive: true

paths:
  # 最终 CVAA 结果保存位置。
  output_root: "/your/cvaa_results"

  # 干净的 RenzKa/simlingo 官方源码目录。
  official_simlingo_root: "/your/RenzKa/simlingo"

  # 官方 SimLingo pytorch_model.pt。
  official_simlingo_checkpoint: "/your/pytorch_model.pt"

  # FLUX.1-Fill-dev 本地模型目录。
  flux_model: "/your/FLUX.1-Fill-dev"

  # LaMa TorchScript 权重。
  lama_model: "/your/big-lama.pt"
```

如果无法从 checkpoint 自动推断 Hydra config，还需要：

```yaml
paths:
  # 官方 SimLingo 训练时保存的 Hydra config。
  official_simlingo_config: "/path/to/config.yaml"
```

---


## 7.1.1 递归查找 route

最终版本的数据输入方式与
`lg_waypoint_planner_project/configs/language_grounded_waypoint.yaml`
保持一致：

```yaml
run:
  # 可以是单条 route，也可以是包含很多 route 的高层父目录。
  input: "/home/.../data/simlingo"

  # true：递归查找所有合法 route。
  # false：直接把 input 本身当作一条 route。
  recursive: true
```

当：

```yaml
recursive: true
```

时，程序不会再依赖：

```text
**/Town*
```

这种文件名规则，而是递归遍历 `run.input` 下的目录，并根据实际内容判断
某个目录是否是一条合法 CARLA route。

合法 route 至少需要包含：

```text
rgb/ 或 rgb_front/
instance_front/
boxes/
measurements/
surround_camera_config.json
```

因此即使未来 route 文件夹名称不是：

```text
Town12_...
```

而是其他名称，只要目录结构正确，仍然可以被发现。

另外，一旦程序发现某个目录已经是一条合法 route，就不会继续递归进入：

```text
rgb/
instance_front/
boxes/
measurements/
```

等内部大目录，因此不会因为递归扫描大量图像文件而显著增加遍历开销。

如果只想调试单条路线：

```yaml
run:
  input: "/path/to/one/route"
  recursive: false
```

此时程序会直接验证该目录是否为合法 route，不再向下搜索。

## 7.2 推荐的低磁盘占用配置

正式大规模实验建议：

```yaml
output:
  # 不永久保存所有反事实图。
  save_counterfactual_images: false

  # 不永久保存所有 mask。
  save_masks: false

  # 不额外保存完整轨迹大文件。
  save_trajectories: false
```

debug 建议：

```yaml
debug:
  # 保留少量 debug，便于确认实验过程没有异常。
  enabled: true

  # 每 20 个 actor 保存一张 debug。
  every_n_actors: 20

  # 保存原图/反事实图 + prediction/GT 对比。
  save_waypoint_comparison: true

  # 正式批量运行时建议关闭 inpainting 五联图，节省磁盘。
  save_inpainting_diagnostic: false
```

runtime：

```yaml
runtime:
  # 当前 chunk 中最多同时保留多少个反事实图。
  max_counterfactuals_per_chunk: 16
```

如果临时磁盘空间非常紧张：

```yaml
runtime:
  max_counterfactuals_per_chunk: 4
```

或者：

```yaml
runtime:
  max_counterfactuals_per_chunk: 8
```

---

## 7.3 当前推荐的正式 inpainting 配置

当前经过前面开发阶段验证的推荐设置：

```yaml
inpainting:
  # LaMa 负责真正删除 actor，FLUX 只负责边界修复。
  backend: "flux_fill"

  # 只对 actor 周围局部区域执行高成本 inpainting。
  local_crop_enabled: true

  # FLUX 只修复 LaMa 结果边界，不重新生成中心区域。
  flux_refine_mode: "seam"

  # 16GB 左右显存建议打开。
  sequential_cpu_offload: true

  # 降低 VAE 峰值显存占用。
  vae_tiling: true
```

这里的核心设计是：

```text
LaMa
=
真正完成目标删除

FLUX
=
只对删除区域边界进行视觉修复
```

而不是：

```text
让 FLUX 重新生成整个 actor 区域
```

这样可以降低 FLUX 再次把原本删除的车辆“幻觉回来”的概率。

---

# 8. 运行方法

全部配置修改完成后：

```bash
conda activate simlingo
```

进入项目：

```bash
cd /path/to/cvaa_baseline_project
```

直接运行：

```bash
python run_pipeline.py
```

正式版本**不需要任何必须的命令行参数**。

---

# 9. 建议先进行 Smoke Test

在正式处理全部数据之前，建议首先修改：

```yaml
data:
  # 只处理 1 条 route。
  max_routes: 1

  # 每条 route 最多处理 2 帧。
  max_frames_per_route: 2
```

debug 设置：

```yaml
debug:
  enabled: true
  every_n_actors: 1
```

这样可以完整检查：

```text
actor-instance matching
mask
LaMa removal
FLUX refinement
Original SimLingo inference
counterfactual inference
GT projection
AD
FD
actor ranking
```

都是否正常。

确认无误后，恢复：

```yaml
data:
  # 0 表示不限制 route 数量。
  max_routes: 0

  # 0 表示不限制 frame 数量。
  max_frames_per_route: 0
```

然后重新：

```bash
python run_pipeline.py
```

即可进行完整批量实验。

---

# 10. 最终输出结构

低磁盘模式下，默认输出：

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
    ├── failures.jsonl
    ├── skipped_interventions.jsonl
    └── debug/
        └── waypoints/
            ├── 0001_actor_3706.jpg
            └── ...
```

---

## `config_used.yaml`

保存本次实验实际使用的配置。

论文最终实验建议一起归档，保证可复现性。

---

## `run_summary.json`

整个批量实验的全局摘要，例如：

```text
处理 route 数量
处理 actor 数量
成功/失败数量
总耗时
模型版本
```

---

## `all_actor_scores.jsonl`

所有 route 的 actor 级结果汇总。

主要包含：

```text
route
frame
actor_id
actor_class
AD
FD
rank
matching score
mask metadata
```

这是后续与论文方法进行统计比较时最重要的文件之一。

---

## `all_actor_scores.csv`

与 JSONL 对应的表格版本，方便：

```text
Excel
Python pandas
Origin
MATLAB
```

直接读取。

---

## `all_frame_rankings.jsonl`

保存每一个 frame 的 actor 排序。

例如：

```text
frame 0001
    rank1 = actor 3707
    rank2 = actor 3706
```

这通常是后续与 LG / 你的关键 actor 结果做 Top-1 一致性比较时最直接的文件。

---

## `<route_id>/summary.json`

当前 route 的处理摘要。

若：

```json
{
  "status": "complete"
}
```

表示该 route 已完整处理完成。

---

## `failures.jsonl`

只有出现失败时才会产生。

用于记录：

```text
frame
actor
失败原因
```

方便定位异常，而不会因为少量 actor 失败导致整个批量实验中断。

---

## `skipped_interventions.jsonl`

记录被过滤掉的 actor，例如：

```text
匹配失败
mask 太小
无法形成有效 intervention
```

这对于论文中报告有效样本数量也有帮助。

---

# 11. Debug 可视化

当：

```yaml
debug:
  enabled: true
  save_waypoint_comparison: true
```

时，会产生 waypoint 对比图。

每个选中的 actor 对应一张左右图：

```text
LEFT
------------------------------------------------
Original front RGB
+
Original SimLingo pred_speed_wps
+
Ground Truth future ego waypoints


RIGHT
------------------------------------------------
Counterfactual front RGB
+
Counterfactual SimLingo pred_speed_wps
+
同一组 Ground Truth future ego waypoints
```

颜色约定：

```text
RED
=
SimLingo predicted future waypoints

GREEN
=
Ground Truth future waypoints
```

点编号：

```text
1
2
3
...
10
```

表示未来不同时刻的位置。

### Ground Truth 是怎样得到的

GT 并不是简单读取某一个 route 字段，而是根据当前帧与未来帧：

```text
ego_matrix
```

按照 Original SimLingo 数据集中的方式转换到当前自车坐标系：

\[
p_t^{ego}
=
R_0^T
(p_t-p_0)
\]

因此：

```text
prediction
和
ground truth
```

处于相同的 ego-frame 坐标系。

需要再次强调：

> Ground Truth 仅用于 debug 可视化，不参与 CVAA 的 AD / FD actor attribution。

---

# 12. Resume / 断点恢复

对于长时间批量实验，推荐：

```yaml
runtime:
  # 已完成 route 在重新运行时直接跳过。
  resume_completed_routes: true

  # 如果 route 上一次未完整结束，则重新构建当前 route 的结果，
  # 避免不同运行版本的数据混在一起。
  rebuild_incomplete_routes: true
```

程序判断 route 是否完成的依据是：

```text
<route_output>/summary.json
```

其中：

```json
{
  "status": "complete"
}
```

则该 route 在下一次运行时被跳过。

### 为什么 incomplete route 默认重建

如果程序中途退出，可能只写了一部分 actor。

如果直接继续 append：

```text
旧配置结果
+
新配置结果
```

可能混在一起。

因此默认采用：

```text
complete route → skip
incomplete route → rebuild
```

更适合作为正式科研实验流程。

临时 chunk 目录在正常结束后会自动删除。

---

# 13. 实现中必须保持的科学实验约束

最终实现围绕以下约束设计。

### 1. 只有 front RGB 发生反事实变化

```text
改变：
front RGB

不改变：
speed
target_point
target_point_next
prompt
camera calibration
checkpoint
network
preprocessing
```

---

### 2. intervention mask 外像素严格不变

正式反事实图会检查：

```text
outside_mask_changed_pixels == 0
```

这样可避免由于图像生成模型对整幅图做无关修改，导致 attribution 失去可解释性。

---

### 3. 原图和反事实图使用同一个 Original SimLingo

网络结构相同：

```text
Original RenzKa/simlingo
```

checkpoint 相同。

---

### 4. 同一 frame 的 original inference 只计算一次

如果当前 frame 有：

```text
actor A
actor B
actor C
```

只需要：

```text
Original inference × 1
Counterfactual inference × 3
```

而不是：

```text
Original inference × 3
```

可以显著减少计算量。

---

### 5. 正式 actor ranking 只使用 `pred_route`

正式：

```text
pred_route
→ AD
→ FD
→ rank
```

而：

```text
pred_speed_wps
```

仅作为辅助诊断和 debug。

---

### 6. Ground Truth 不参与 attribution

GT future waypoints：

```text
只用于可视化
```

不会影响：

```text
AD
FD
rank
```

---

# 14. 正式论文实验建议流程

在运行全部数据之前，推荐使用：

```yaml
data:
  max_routes: 1
  max_frames_per_route: 2

debug:
  enabled: true
  every_n_actors: 1
```

逐项确认：

1. actor mask 是否对应正确目标；
2. 删除后是否确实移除了目标 actor；
3. 是否没有误删其他交通参与者；
4. 路面背景是否合理；
5. `outside_mask_changed_pixels == 0`；
6. Original SimLingo 是否正常输出；
7. prediction / GT debug 是否投影正确；
8. AD / FD 是否生成；
9. actor ranking 是否合理。

全部验证通过后：

```yaml
data:
  max_routes: 0
  max_frames_per_route: 0
```

开始正式全量实验。

对于大规模数据，可以降低 debug 保存频率：

```yaml
debug:
  every_n_actors: 20
```

或者完全关闭：

```yaml
debug:
  enabled: false
```

关闭 debug：

> 不会改变正式 CVAA 的 AD / FD 结果。

---

# 15. 轻量自测试

项目包含基础单元测试：

```bash
python -m unittest discover -s tests
```

当前主要验证：

```text
AD 计算
FD 计算
AD → FD 排序逻辑
adaptive mask
mask 外像素严格保持不变
```

这些测试不需要加载完整：

```text
FLUX
LaMa
SimLingo
CARLA dataset
```

因此可以快速检查项目核心数值逻辑。

完整 end-to-end 实验仍然需要准备前面说明的模型和数据。

---

# 16. 可复现性建议

论文最终发布时，建议同时保存：

```text
config_used.yaml
Original SimLingo Git commit
Original SimLingo checkpoint checksum
FLUX model version
big-lama.pt checksum
CARLA 数据生成代码 commit
当前 CVAA baseline project commit
```

这样后续可以准确回答：

```text
当时使用了哪一个 SimLingo？
哪一个 FLUX？
哪一个 LaMa？
哪一版数据？
哪一组配置？
```

对于论文复现尤其重要。

---

# 17. 第三方依赖与许可证

第三方项目和模型来源请参见：

```text
THIRD_PARTY_NOTICES.md
```

包括：

```text
Original SimLingo
FLUX.1-Fill-dev
LaMa
CARLA
```

等。

需要注意：

> 当前项目仓库本身的 LICENSE 需要由项目作者在公开 GitHub 前自行选择。

例如可以根据实际发布需求选择：

```text
MIT
Apache-2.0
GPL
```

等。

无论本项目选择何种 LICENSE：

```text
第三方源码和第三方模型仍然受其自身许可证约束
```

不能被本项目 LICENSE 覆盖。

---

# 18. 最简使用流程

如果已经完成环境和模型准备，真正运行时只需要三步。

### 第一步：修改配置

```bash
vim config.yaml
```

至少填写：

```yaml
paths:
  routes_root: "..."
  output_root: "..."
  official_simlingo_root: "..."
  official_simlingo_checkpoint: "..."
  flux_model: "..."
  lama_model: "..."
```

---

### 第二步：激活环境

```bash
conda activate simlingo
```

---

### 第三步：启动

```bash
python run_pipeline.py
```

随后程序会自动完成：

```text
route discovery
→ actor-instance matching
→ mask construction
→ visual counterfactual generation
→ Original SimLingo inference
→ AD / FD
→ actor ranking
→ compact result saving
```

无需再手工执行 Stage 1 / Stage 2 / Stage 3 / Stage 4。

---

# 19. 一句话总结

该项目最终实现的是：

> **在 CARLA 仿真真值支持下，对场景中的每一个候选交通参与者进行视觉删除，以原始官方 SimLingo 对删除前后的驾驶响应变化进行测量，并依据 AD / FD 对交通参与者进行反事实重要性排序，同时通过分块、临时缓存和按需 debug 控制本地磁盘与 GPU 资源占用。**



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

