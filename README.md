# CVAA Baseline 构建与部署说明

本项目用于在 SimLingo/CARLA 数据集上运行 CVAA baseline，并通过 `keyframes.txt` 直接读取统一关键帧完成正式实验。

## 1. 获取项目

```bash
git clone https://github.com/liuly7002/CVAA_baseline.git
cd CVAA_baseline
```

## 2. 准备 Conda 环境

项目使用两个独立环境：

- `simlingo`：运行官方 SimLingo 推理
- `cvaa_fill`：运行 LaMa + FLUX 反事实图像生成

### 2.1 SimLingo 环境

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

### 2.2 cvaa_fill 环境

`cvaa_fill` 环境专门用于运行 LaMa 和 FLUX.1-Fill-dev，负责生成删除交通参与者后的反事实图像。该环境与 `simlingo` 环境相互独立，不需要在其中安装 SimLingo。

创建 Conda 环境：

```bash
conda create -n cvaa_fill python=3.10 -y
```

激活环境：

```bash
conda activate cvaa_fill
```

安装 PyTorch。当前项目使用 NVIDIA GPU，推荐安装 CUDA 12.1 对应版本：

```bash
先用阿里云安装 Pillow 及其他库：
pip install pillow -i https://mirrors.aliyun.com/pypi/simple/

pip install \
  pillow \
  numpy==1.26.4 \
  scipy==1.13.1 \
  filelock \
  typing-extensions \
  sympy \
  networkx \
  jinja2 \
  fsspec \
  requests \
  -i https://mirrors.aliyun.com/pypi/simple/

然后安装其他的：
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
```

安装 FLUX 所需依赖：

```bash
pip install diffusers==0.32.2 transformers==4.46.3 accelerate==1.0.1
```

安装完成后检查 PyTorch 和 GPU：

```bash
python -c "import torch; print('Torch:', torch.__version__); print('CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

正常情况下应能够看到类似：

```text
Torch: 2.2.0+cu121
CUDA: 12.1
CUDA available: True
GPU: NVIDIA GeForce RTX 4090
```

检查 Diffusers 和 `FluxFillPipeline` 是否可以正常导入：

```bash
python -c "import diffusers, transformers, accelerate; from diffusers import FluxFillPipeline; print('diffusers:', diffusers.__version__); print('transformers:', transformers.__version__); print('accelerate:', accelerate.__version__); print('FluxFillPipeline import OK')"
```

正常情况下应输出：

```text
diffusers: 0.32.2
transformers: 4.46.3
accelerate: 1.0.1
FluxFillPipeline import OK
```

再检查 LaMa 所需的基础依赖：

```bash
python -c "import torch, cv2, numpy, PIL; print('LaMa dependencies OK')"
```

如果提示缺少 `opencv-python`、`numpy` 或 `Pillow`，安装：

```bash
pip install opencv-python==4.10.0.84 --no-deps -i https://mirrors.aliyun.com/pypi/simple/
```

环境配置完成后，可以通过下面的命令进行最终检查：

```bash
conda activate cvaa_fill

python -c "import torch, cv2, numpy, diffusers, transformers, accelerate; from diffusers import FluxFillPipeline; print('Python environment OK'); print('Torch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0)); print('Diffusers:', diffusers.__version__)"
```

只要能够正常识别 GPU，并且 `FluxFillPipeline` 可以成功导入，即说明 `cvaa_fill` 环境已经构建完成。

正式运行 CVAA 时不需要手动进入 `cvaa_fill` 环境。正常情况下只需要：

```bash
conda activate simlingo
python run_pipeline.py
```

主程序会根据 `config.yaml` 中：

```yaml
environments:
  simlingo_conda_env: "simlingo"
  cvaa_fill_conda_env: "cvaa_fill"
```

自动找到两个 Conda 环境对应的 Python，并分别启动 SimLingo worker 和 LaMa + FLUX worker。


## 3. 准备模型

需要准备以下模型。

### 3.1 官方 SimLingo

需要官方 SimLingo 源码、checkpoint 和对应 Hydra 配置。


### 3.2 FLUX.1-Fill-dev


### 3.3 LaMa


## 4. 准备关键帧文件

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

## 5. 修改 config.yaml

进入项目：

```bash
cd /home/kemove/ll/CVAA_baseline
```

编辑：

```bash
vim config.yaml
```

需要重点检查以下配置。

### 5.1 数据集根目录

```yaml
run:
  input: "/你的数据集根目录/data/simlingo"
```

例如：

```yaml
run:
  input: "/home/kemove/ll/simlingo_liulei/database/simlingo_v2_2026_09_12_22_24_28/data/simlingo"
```

### 5.2 关键帧文件

```yaml
data:
  keyframe_file: "/你的数据集根目录/data/simlingo/keyframes.txt"
```

例如：

```yaml
data:
  keyframe_file: "/home/kemove/ll/simlingo_liulei/database/simlingo_v2_2026_09_12_22_24_28/data/simlingo/keyframes.txt"
```

### 5.3 输出目录

```yaml
paths:
  output_root: "/你的数据集根目录/data/simlingo/cvaa_results"
```

### 5.4 SimLingo 路径

```yaml
paths:
  official_simlingo_root: "/home/kemove/models/simlingo"
  official_simlingo_checkpoint: "/home/kemove/models/simlingo/checkpoint/simlingo/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt"
  official_simlingo_config: null
```

如果 checkpoint 目录中可以自动找到 `.hydra/config.yaml`，`official_simlingo_config` 保持 `null`。

### 5.5 FLUX 和 LaMa 路径

```yaml
paths:
  flux_model: "/home/kemove/models/FLUX.1-Fill-dev"
  lama_model: "/home/kemove/models/LAMA/big-lama.pt"
```

### 5.6 双 GPU 设置

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

### 5.7 FLUX 显存配置

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

## 6. 小规模测试

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

## 7. 检查双 GPU

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

## 8. 正式运行

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

## 9. 输出结果

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

## 10. 中断后继续运行

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
