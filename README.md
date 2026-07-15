# 卷积神经网络与 ViT 结合的探索与实现

本实验围绕 CIFAR-10 图像分类任务，分别实现并比较 Tiny CNN、Tiny ViT、CNN-ViT 混合模型以及两组消融模型。实验重点是观察卷积神经网络的局部特征提取能力与 Vision Transformer 的全局建模能力结合后，对分类准确率、训练代价和推理效率的影响。

## 实验环境

实验依赖的主要环境如下：

| 项目 | 配置 |
| --- | --- |
| 操作系统 | Windows 10 |
| Python | Python 3.10.20 |
| 深度学习框架 | PyTorch 2.11.0 + CUDA 12.6 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| 数据集 | CIFAR-10 |
| 主要依赖 | torch、torchvision、numpy、matplotlib、tqdm |

安装依赖：

```bash
pip install -r requirements.txt
```

如果使用 CPU 运行，可以正常安装 CPU 版本 PyTorch；如果使用 NVIDIA GPU，建议根据本机 CUDA 版本安装对应的 PyTorch GPU 版本。

## 数据集下载

本实验使用 CIFAR-10 数据集。CIFAR-10 包含 60000 张 32×32 彩色图像，共 10 个类别，其中训练集 50000 张，测试集 10000 张。类别包括 airplane、automobile、bird、cat、deer、dog、frog、horse、ship、truck。

数据集官方下载页面：

```text
https://www.cs.toronto.edu/~kriz/cifar.html
```

本项目代码已经支持通过 torchvision 自动下载 CIFAR-10。首次运行时在命令中加入 `--download` 即可，数据会保存到 `--data-root` 指定的目录中，例如 `./data`。

## 运行方式

先进入代码目录：

```bash
cd repro_code
```

安装依赖：

```bash
pip install -r requirements.txt
```

快速检查环境是否可运行：

```bash
python run_experiments.py --dataset cifar10 --data-root ./data --download --quick
```

正式运行 5 组实验：

```bash
python run_experiments.py --dataset cifar10 --data-root ./data --download --epochs 100 --batch-size 128 --device cuda
```

如果没有 NVIDIA GPU，可以改为 CPU：

```bash
python run_experiments.py --dataset cifar10 --data-root ./data --download --epochs 100 --batch-size 128 --device cpu --no-amp
```

如果只想单独训练主模型，可以运行：

```bash
python train.py --model hybrid_tiny --dataset cifar10 --data-root ./data --download --patch-size 2 --epochs 100 --batch-size 128 --device cuda
```

`run_experiments.py` 会依次运行以下 5 个模型：

| 序号 | 模型 | 说明 |
| --- | --- | --- |
| 1 | cnn_tiny | 纯 CNN 基线模型 |
| 2 | vit_tiny | 纯 ViT 基线模型 |
| 3 | hybrid_tiny, patch=2 | CNN-ViT 主模型 |
| 4 | hybrid_tiny, patch=4 | patch size 消融实验 |
| 5 | hybrid_no_fusion | 去除融合模块的消融实验 |

## 实验结果

本实验在 CIFAR-10 上对 5 个模型均训练 100 epoch，得到结果如下：

| 模型 | 参数量 | 最佳 epoch | 验证损失 | 验证准确率 | 单张推理延迟 | 训练耗时 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Tiny CNN | 127,018 | 100 | 0.770831 | 89.46% | 7.64 ms | 141.98 min |
| Tiny ViT | 1,205,898 | 88 | 0.973719 | 80.35% | 9.49 ms | 147.08 min |
| Hybrid p=2 | 1,007,242 | 92 | 0.745805 | 91.36% | 21.74 ms | 178.46 min |
| Hybrid p=4 | 1,007,242 | 98 | 0.749011 | 91.33% | 20.40 ms | 177.89 min |
| Hybrid no-fusion | 725,738 | 95 | 0.745960 | 91.09% | 18.81 ms | 175.90 min |

从实验结果可以看出，CNN-ViT 混合模型取得了最高验证准确率。其中 Hybrid p=2 的验证准确率为 91.36%，高于 Tiny CNN 的 89.46% 和 Tiny ViT 的 80.35%。这说明在 CIFAR-10 分类任务中，卷积模块提供的局部特征提取能力与 Transformer 模块提供的全局关系建模能力具有互补作用。

同时，混合模型的推理延迟和训练耗时也高于纯 CNN，说明模型精度提升是以更高计算代价为前提的。消融实验中，Hybrid p=4 与 Hybrid p=2 准确率接近，但延迟略低；Hybrid no-fusion 的准确率低于主模型，说明局部特征与全局特征的融合模块对最终分类效果有一定帮助。
