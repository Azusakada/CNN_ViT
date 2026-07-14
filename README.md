# 卷积神经网络与 ViT 结合的探索与实现

本项目实现并统一评测三类图像分类模型：轻量 CNN、Tiny ViT，以及受 MobileViT 启发的 CNN-ViT 混合模型。混合块先用卷积提取局部特征，再把同一相对位置的 patch 像素组织成序列，用 Transformer 建模跨区域依赖，最后折叠并与原特征融合。

## 1. 环境

推荐 Python 3.10 或 3.11，CUDA 11.8/12.x 均可。先进入本目录：

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## 2. 数据集

### CIFAR-10 / CIFAR-100

首次运行添加 `--download`，程序会通过 torchvision 下载到 `--data-root`。

### 自定义数据集

使用 torchvision `ImageFolder` 格式：

```text
dataset_root/
  train/
    class_a/*.jpg
    class_b/*.jpg
  val/
    class_a/*.jpg
    class_b/*.jpg
```

训练时指定 `--dataset imagefolder --data-root /path/to/dataset_root`。

## 3. 先做快速自检

下面的命令只跑 1 个 epoch、每阶段 2 个 batch，用于检查环境和数据路径，不可作为报告结果：

```bash
python run_experiments.py --dataset cifar10 --data-root ./data --download --quick
```

## 4. 正式实验

建议先在 CIFAR-10 上运行 100 epoch 的完整对比与消融：

```bash
python run_experiments.py \
  --dataset cifar10 \
  --data-root ./data \
  --download \
  --epochs 100 \
  --batch-size 128 \
  --device cuda
```

Windows PowerShell 可写成一行：

```powershell
python run_experiments.py --dataset cifar10 --data-root .\data --download --epochs 100 --batch-size 128 --device cuda
```

脚本依次运行：

1. `cnn_tiny`：纯卷积基线；
2. `vit_tiny`：纯 Transformer 基线；
3. `hybrid_tiny, patch=2`：主模型；
4. `hybrid_tiny, patch=4`：patch size 消融；
5. `hybrid_no_fusion`：去除局部/全局融合的消融。

如显存不足，将 `--batch-size` 改为 64 或 32。无 NVIDIA GPU 时可用 `--device cpu --no-amp`；Apple Silicon 可用 `--device mps --no-amp`。

## 5. 单个实验与调参

```bash
python train.py \
  --model hybrid_tiny \
  --dataset cifar10 \
  --data-root ./data \
  --download \
  --patch-size 2 \
  --optimizer adamw \
  --lr 3e-4 \
  --weight-decay 0.05 \
  --epochs 100 \
  --batch-size 128 \
  --seed 42 \
  --device cuda
```

需要从中断处继续时：

```bash
python train.py ... --resume runs/某次实验/last.pt
```

为了让对比公平，请保持数据划分、随机种子、epoch、优化器和增强策略一致。若时间允许，可将正式对比用种子 42、123、2026 各运行一次，并报告均值与标准差。

## 6. 输出文件

每次实验目录包含：

- `config.json`、`environment.json`：参数与软硬件环境；
- `best.pt`、`last.pt`：最佳与最近权重；
- `metrics.csv`、`curves.png`：逐 epoch 指标和训练曲线；
- `confusion_matrix.csv/png`：混淆矩阵；
- `summary.json`：最佳准确率、参数量、batch=1 推理延迟等。

完整实验结束后，`runs/experiment_summary.csv` 汇总全部模型。请把该 CSV、各实验的 `summary.json` 和 `curves.png` 发回，即可自动回填技术报告中的表格、图和结论。

## 7. 复现注意事项

- 正式实验不要使用 `--quick` 或 batch 限制参数。
- 推理延迟依赖硬件，只能比较在同一台设备、同一软件环境下测得的结果。
- 若更改输入分辨率，所有模型必须使用同一 `--image-size`。
- `--deterministic` 可提高严格复现性，但可能降低训练速度。
- 提交前保留源码、README 和最终 `experiment_summary.csv`；权重体积较大时可不上传或仅上传最佳权重。

