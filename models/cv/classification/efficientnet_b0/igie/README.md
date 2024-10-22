# EfficientNet B0

## Description

EfficientNet-B0 is a lightweight yet highly efficient convolutional neural network architecture. It is part of the EfficientNet family, known for its superior performance in balancing model size and accuracy. Developed with a focus on resource efficiency, EfficientNet-B0 achieves remarkable results across various computer vision tasks, including image classification and feature extraction.

## Setup

### Install

```bash
pip3 install -r requirements.txt
```

### Download

Pretrained model: <https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Model Conversion

```bash
python3 export.py --weight efficientnet_b0_rwightman-7f5810bc.pth --output efficientnet_b0.onnx
```

## Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
```

### FP16

```bash
# Accuracy
bash scripts/infer_efficientnet_b0_fp16_accuracy.sh
# Performance
bash scripts/infer_efficientnet_b0_fp16_performance.sh
```

## Results

Model           |BatchSize  |Precision |FPS       |Top-1(%)  |Top-5(%)
----------------|-----------|----------|----------|----------|--------
EfficientNet_B0 |    32     |   FP16   | 2596.60  |  77.639  | 93.540
