# CVAA Baseline 构建与部署说明

本项目用于在 SimLingo/CARLA 数据集上运行 CVAA baseline，并通过 `keyframes.txt` 直接读取统一关键帧完成正式实验。

## 1. 获取项目

```bash
git clone https://github.com/liuly7002/CVAA_baseline.git
cd CVAA_baseline
```

如果项目已经存在，直接进入项目目录：

```bash
cd /home/kemove/ll/CVAA_baseline
```

## 2. 应用优化后的文件

将优化包解压后，在优化包目录执行：

```bash
bash apply_to_repo.sh /home/kemove/ll/CVAA_baseline
```

成功后会看到：

```text
Applied optimized CVAA files to: /home/kemove/ll/CVAA_baseline
```

然后进入项目：

```bash
cd /home/kemove/ll/CVAA_baseline
```

## 3. 准备 Conda 环境

项目使用两个独立环境：

- `simlingo`：运行官方 SimLingo 推理
- `cvaa_fill`：运行 LaMa + FLUX 反事实图像生成

### 3.1 SimLingo 环境

激活环境：

```bash
conda activate simlingo
```

检查关键依赖：

```bash
python -c "import torch, transformers, accelerate; print(torch.__version__)"
```

检查 GPU：

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### 3.2 cvaa_fill 环境

激活环境：

```bash
conda activate cvaa_fill
```

安装依赖：

```bash
pip install diffusers==0.32.2 transformers==4.46.3 accelerate==1.0.1
```

检查 FLUX 相关依赖：

```bash
python -c "from diffusers import FluxFillPipeline; import torch; print(torch.cuda.is_available())"
```

## 4. 准备模型

需要准备以下模型。

### 4.1 官方 SimLingo

需要官方 SimLingo 源码、checkpoint 和对应 Hydra 配置。

例如：

```text
/home/kemove/models/simlingo
/home/kemove/models/simlingo/checkpoint/simlingo/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt
```

### 4.2 FLUX.1-Fill-dev

例如：

```text
/home/kemove/models/FLUX.1-Fill-dev
```

### 4.3 LaMa

需要：

```text
/home/kemove/models/LAMA/big-lama.pt
```

## 5. 准备关键帧文件

正式实验直接读取：

```text
keyframes.txt
```

每一行格式为：

```text
相对数据集根目录的 route 路径/帧号
```

例如：

```text
lb1_split/routes_training/noScenarios/route_xxx/000123
```

该文件应由独立关键帧筛选程序提前生成。

## 6. 修改 config.yaml

进入项目：

```bash
cd /home/kemove/ll/CVAA_baseline
```

编辑：

```bash
vim config.yaml
```

需要重点检查以下配置。

### 6.1 数据集根目录

```yaml
run:
  input: "/你的数据集根目录/data/simlingo"
```

例如：

```yaml
run:
  input: "/home/kemove/ll/simlingo_liulei/database/simlingo_v2_2026_09_12_22_24_28/data/simlingo"
```

### 6.2 关键帧文件

```yaml
data:
  keyframe_file: "/你的数据集根目录/data/simlingo/keyframes.txt"
```

例如：

```yaml
data:
  keyframe_file: "/home/kemove/ll/simlingo_liulei/database/simlingo_v2_2026_09_12_22_24_28/data/simlingo/keyframes.txt"
```

### 6.3 输出目录

```yaml
paths:
  output_root: "/你的数据集根目录/data/simlingo/cvaa_results"
```

### 6.4 SimLingo 路径

```yaml
paths:
  official_simlingo_root: "/home/kemove/models/simlingo"
  official_simlingo_checkpoint: "/home/kemove/models/simlingo/checkpoint/simlingo/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt"
  official_simlingo_config: null
```

如果 checkpoint 目录中可以自动找到 `.hydra/config.yaml`，`official_simlingo_config` 保持 `null`。

### 6.5 FLUX 和 LaMa 路径

```yaml
paths:
  flux_model: "/home/kemove/models/FLUX.1-Fill-dev"
  lama_model: "/home/kemove/models/LAMA/big-lama.pt"
```

### 6.6 双 GPU 设置

两张 RTX 4090 时保持：

```yaml
runtime:
  inpainting_gpu: 0
  simlingo_gpu: 1
```

其中：

```text
GPU 0：LaMa + FLUX
GPU 1：Official SimLingo
```

### 6.7 FLUX 显存配置

默认：

```yaml
inpainting:
  lama_device: "cuda"
  cpu_offload: true
  sequential_cpu_offload: false
```

如果运行时出现 CUDA OOM，可改为：

```yaml
inpainting:
  cpu_offload: false
  sequential_cpu_offload: true
```

其余正式实验参数不需要修改。

## 7. 小规模测试

正式运行前，先设置：

```yaml
data:
  max_routes: 3
```

激活 SimLingo 环境：

```bash
conda activate simlingo
```

进入项目：

```bash
cd /home/kemove/ll/CVAA_baseline
```

运行：

```bash
python run_pipeline.py
```

启动后应看到类似：

```text
CVAA optimized key-frame pipeline
manifest routes: ...
manifest keyframes: ...
GPU0 fill / GPU1 SimLingo: 0 / 1
```

模型初始化阶段应看到：

```text
[LaMa] loading ...
[FLUX] loading ...
[OFFICIAL SimLingo] loading checkpoint ...
```

整个任务中这些模型只应加载一次。

## 8. 检查双 GPU

另开一个终端：

```bash
watch -n 1 nvidia-smi
```

正常情况下：

```text
GPU 0：运行 LaMa + FLUX
GPU 1：运行 SimLingo
```

进入流水阶段后，两张 GPU 都应有计算任务。

## 9. 正式运行

小规模测试通过后，将：

```yaml
data:
  max_routes: 3
```

改回：

```yaml
data:
  max_routes: 0
```

然后执行：

```bash
conda activate simlingo
cd /home/kemove/ll/CVAA_baseline
python run_pipeline.py
```

程序会读取完整 `keyframes.txt` 并处理全部关键帧。

## 10. 输出结果

结果保存在：

```text
paths.output_root
```

主要文件：

```text
cvaa_results/
├── all_actor_scores.jsonl
├── all_actor_scores.csv
├── all_frame_rankings.jsonl
├── run_summary.json
├── benchmark_identity.json
├── config_used.yaml
└── 各 route 结果目录/
```

其中：

- `all_actor_scores.jsonl`：所有 actor 的 AD、FD 等结果
- `all_frame_rankings.jsonl`：每个关键帧内 actor 的最终排序
- `run_summary.json`：整个实验运行统计
- `benchmark_identity.json`：记录关键帧文件 SHA256 和实验签名
- `config_used.yaml`：记录本次实际使用的配置

## 11. 中断后继续运行

默认配置：

```yaml
runtime:
  resume_completed_routes: true
  rebuild_incomplete_routes: true
```

如果运行中断，直接再次执行：

```bash
python run_pipeline.py
```

已经完成且配置一致的 route 会自动跳过，未完成的 route 会重新处理。
