# DenseNet

## Description

Dense Convolutional Network (DenseNet), connects each layer to every other layer in a feed-forward fashion. Whereas traditional convolutional networks with L layers have L connections - one between each layer and its subsequent layer - our network has L(L+1)/2 direct connections.

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

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Model Conversion

```bash
mkdir checkpoints
python3 export_onnx.py --output_model checkpoints/densenet121.onnx
```

## Inference

```bash
export DATASETS_DIR=/path/to/imagenet_val/
export CHECKPOINTS_DIR=./checkpoints
export RUN_DIR=./
export CONFIG_DIR=config/DENSENET121_CONFIG
```

### FP16

```bash
# Accuracy
bash scripts/infer_densenet_fp16_accuracy.sh
# Performance
bash scripts/infer_densenet_fp16_performance.sh
```

## Results

| Model    | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
| -------- | --------- | --------- | ------- | -------- | -------- |
| DenseNet | 32        | FP16      | 1536.89 | 0.7442   | 0.9197   |
