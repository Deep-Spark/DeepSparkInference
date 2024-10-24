# EfficientNet B1

## Description

EfficientNet B1 is a convolutional neural network architecture that falls under the EfficientNet family, known for its remarkable balance between model size and performance. Introduced as part of the EfficientNet series, EfficientNet B1 offers a compact yet powerful solution for various computer vision tasks, including image classification, object detection and segmentation.

## Setup

### Install

```bash
pip3 install -r requirements.txt
```

### Download

Pretrained model: <https://download.pytorch.org/models/efficientnet_b1-c27df63c.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Model Conversion

```bash
python3 export.py --weight efficientnet_b1-c27df63c.pth --output efficientnet_b1.onnx
```

## Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
```

### FP16

```bash
# Accuracy
bash scripts/infer_efficientnet_b1_fp16_accuracy.sh
# Performance
bash scripts/infer_efficientnet_b1_fp16_performance.sh
```

## Results

Model           |BatchSize  |Precision |FPS      |Top-1(%) |Top-5(%)
----------------|-----------|----------|---------|---------|--------
EfficientNet B1 |    32     |   FP16   | 1292.31 | 78.823  | 94.494
