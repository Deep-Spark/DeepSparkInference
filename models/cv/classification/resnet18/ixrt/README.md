# Resnet18

## Description

ResNet-18 is a variant of the ResNet (Residual Network) architecture, which was introduced by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun in their 2016 paper, "Deep Residual Learning for Image Recognition." The ResNet architecture was pivotal in addressing the challenges of training very deep neural networks by introducing residual blocks.

## Setup

### Install

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-dev

pip3 install tqdm
pip3 install onnx
pip3 install onnxsim
pip3 install tabulate
```

### Download

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Model Conversion

```bash
mkdir checkpoints
python3 export_onnx.py --output_model checkpoints/resnet18.onnx
```

## Inference

```bash
export PROJ_DIR=./
export DATASETS_DIR=/path/to/imagenet_val/
export CHECKPOINTS_DIR=./checkpoints
export RUN_DIR=./
export CONFIG_DIR=config/RESNET18_CONFIG
```

### FP16

```bash
# Accuracy
bash scripts/infer_resnet18_fp16_accuracy.sh
# Performance
bash scripts/infer_resnet18_fp16_performance.sh
```

### INT8

```bash
# Accuracy
bash scripts/infer_resnet18_int8_accuracy.sh
# Performance
bash scripts/infer_resnet18_int8_performance.sh
```

## Results

Model    |BatchSize  |Precision |FPS       |Top-1(%)  |Top-5(%)
---------|-----------|----------|----------|----------|--------
Resnet18 |    32     |   FP16   | 9592.98  |  69.77   | 89.09
Resnet18 |    32     |   INT8   | 21314.55 |  69.53   | 88.97
