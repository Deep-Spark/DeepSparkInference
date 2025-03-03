# ResNet34

## Description

Residual Networks, or ResNets, learn residual functions with reference to the layer inputs, instead of learning unreferenced functions. Instead of hoping each few stacked layers directly fit a desired underlying mapping, residual nets let these layers fit a residual mapping.

## Setup

### Install

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

pip3 install -r requirements.txt
```

### Download

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Model Conversion

```bash
mkdir checkpoints
python3 export_onnx.py --output_model checkpoints/resnet34.onnx
```

## Inference

```bash
export PROJ_DIR=./
export DATASETS_DIR=/path/to/imagenet_val/
export CHECKPOINTS_DIR=./checkpoints
export RUN_DIR=./
export CONFIG_DIR=config/RESNET34_CONFIG
```

### FP16

```bash
# Accuracy
bash scripts/infer_resnet34_fp16_accuracy.sh
# Performance
bash scripts/infer_resnet34_fp16_performance.sh
```

### INT8

```bash
# Accuracy
bash scripts/infer_resnet34_int8_accuracy.sh
# Performance
bash scripts/infer_resnet34_int8_performance.sh
```

## Results

Model    |BatchSize  |Precision |FPS       |Top-1(%)  |Top-5(%)
---------|-----------|----------|----------|----------|--------
ResNet34 |    32     |   FP16   | 6179.47  |  73.30   | 91.42
ResNet34 |    32     |   INT8   | 11256.36 |  73.13   | 91.34
