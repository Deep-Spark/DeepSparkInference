# MobileNetV3_Small

## Description

MobileNetV3_Small is a lightweight convolutional neural network architecture designed for efficient mobile and embedded devices. It is part of the MobileNet family, renowned for its compact size and high performance, making it ideal for applications with limited computational resources.The key focus of MobileNetV3_Small is to achieve a balance between model size, speed, and accuracy.

## Setup

### Install

```bash
pip3 install -r requirements.txt
```

### Download

Pretrained model: <https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Model Conversion

```bash
python3 export.py --weight mobilenet_v3_small-047dcff4.pth --output mobilenetv3_small.onnx
```

## Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
```

### FP16

```bash
# Accuracy
bash scripts/infer_mobilenet_v3_fp16_accuracy.sh
# Performance
bash scripts/infer_mobilenet_v3_fp16_performance.sh
```

## Results

Model             |BatchSize  |Precision |FPS      |Top-1(%) |Top-5(%)
------------------|-----------|----------|---------|---------|--------
MobileNetV3_Small |    32     |   FP16   | 6837.86 | 67.612  | 87.404
