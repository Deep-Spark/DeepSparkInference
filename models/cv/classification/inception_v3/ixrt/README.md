# Inception V3

## Description

Inception v3 is a convolutional neural network architecture designed for image recognition and classification tasks. Developed by Google, it represents an evolution of the earlier Inception models. Inception v3 is characterized by its deep architecture, featuring multiple layers with various filter sizes and efficient use of computational resources. The network employs techniques like factorized convolutions and batch normalization to enhance training stability and accelerate convergence.

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

Pretrained model: <https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Model Conversion

```bash

mkdir checkpoints
python3 export.py --weight inception_v3_google-0cc3c7bd.pth --output checkpoints/inception_v3.onnx
```

## Inference

```bash
export PROJ_DIR=/Path/to/inception_v3/ixrt
export DATASETS_DIR=/path/to/imagenet_val/
export CHECKPOINTS_DIR=./checkpoints
export RUN_DIR=/Path/to/inception_v3/ixrt
export CONFIG_DIR=/Path/to/config/INCEPTION_V3_CONFIG
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```

### FP16

```bash
# Accuracy
bash scripts/infer_inception_v3_fp16_accuracy.sh
# Performance
bash scripts/infer_inception_v3_fp16_performance.sh
```

### INT8

```bash
# Accuracy
bash scripts/infer_inception_v3_int8_accuracy.sh
# Performance
bash scripts/infer_inception_v3_int8_performance.sh
```

## Results

Model        |BatchSize  |Precision |FPS       |Top-1(%)  |Top-5(%)
-------------|-----------|----------|----------|----------|--------
Inception_v3 |    32     |   FP16   | 3515.29  |  70.64   | 89.33
Inception_v3 |    32     |   INT8   | 4916.32  |  70.45   | 89.28
