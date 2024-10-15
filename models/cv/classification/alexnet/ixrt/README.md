# AlexNet

## Description

AlexNet is a classic convolutional neural network architecture. It consists of convolutions, max pooling and dense layers as the basic building blocks.

## Setup

### Install

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-dev

pip3 install -r requirements.txt
```

### Download

Pretrained model: <https://download.pytorch.org/models/alexnet-owt-7be5be79.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Model Conversion

```bash
mkdir checkpoints
python3 export_onnx.py --origin_model /path/to/alexnet-owt-7be5be79.pth --output_model checkpoints/alexnet.onnx
```

## Inference

```bash
export PROJ_DIR=./
export DATASETS_DIR=/path/to/imagenet_val/
export CHECKPOINTS_DIR=./checkpoints
export RUN_DIR=./
export CONFIG_DIR=config/ALEXNET_CONFIG
```

### FP16

```bash
# Accuracy
bash scripts/infer_alexnet_fp16_accuracy.sh
# Performance
bash scripts/infer_alexnet_fp16_performance.sh
```

### INT8

```bash
# Accuracy
bash scripts/infer_alexnet_int8_accuracy.sh
# Performance
bash scripts/infer_alexnet_int8_performance.sh
```

## Results

Model   |BatchSize  |Precision |FPS       |Top-1(%)  |Top-5(%)
--------|-----------|----------|----------|----------|--------
AlexNet |    32     |   FP16   | 17644.90 |  56.54   | 79.08
AlexNet |    32     |   INT8   | 18276.83 |  55.37   | 79.04
