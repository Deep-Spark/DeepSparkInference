# ResNetV1D50

## Description

Residual Networks, or ResNets, learn residual functions with reference to the layer inputs, instead of learning unreferenced functions. Instead of hoping each few stacked layers directly fit a desired underlying mapping, residual nets let these layers fit a residual mapping.

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
pip3 install ppq
pip3 install mmpretrain
pip3 install mmcv-lite
```

### Download

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Model Conversion

```bash
mkdir checkpoints
python3 export_onnx.py --output_model checkpoints/resnet_v1_d50.onnx
```

## Inference

```bash
export DATASETS_DIR=/path/to/imagenet_val/
export CHECKPOINTS_DIR=./checkpoints
export RUN_DIR=./
export CONFIG_DIR=config/RESNET_V1_D50_CONFIG
```

### FP16

```bash
# Accuracy
bash scripts/infer_resnet_v1_d50_fp16_accuracy.sh
# Performance
bash scripts/infer_resnet_v1_d50_fp16_performance.sh
```

### INT8

```bash
# Accuracy
bash scripts/infer_resnet_v1_d50_int8_accuracy.sh
# Performance
bash scripts/infer_resnet_v1_d50_int8_performance.sh
```

## Results

| Model         | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
| ------------- | --------- | --------- | ------- | -------- | -------- |
| ResNet_V1_D50 | 32        | FP16      | 3887.55 | 0.77544  | 0.93568  |
| ResNet_V1_D50 | 32        | INT8      | 7148.58 | 0.7711   | 0.93514  |
