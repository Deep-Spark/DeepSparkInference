# SqueezeNet1_0

## Description

SqueezeNet1_0 is a lightweight convolutional neural network introduced in the paper "SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size." It was designed to achieve high classification accuracy with significantly fewer parameters, making it highly efficient for resource-constrained environments.The core innovation of SqueezeNet lies in the Fire Module, which reduces parameters using 1x1 convolutions in the "Squeeze layer" and expands feature maps through a mix of 1x1 and 3x3 convolutions in the "Expand layer." Additionally, delayed downsampling improves feature representation and accuracy.

## Setup

### Install

```bash
pip3 install -r requirements.txt
```

### Download

Pretrained model: <https://download.pytorch.org/models/squeezenet1_0-b66bff10.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Model Conversion

```bash
python3 export.py --weight squeezenet1_0-b66bff10.pth --output squeezenet1_0.onnx
```

## Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
```

### FP16

```bash
# Accuracy
bash scripts/infer_squeezenet1_0_fp16_accuracy.sh
# Performance
bash scripts/infer_squeezenet1_0_fp16_performance.sh
```

## Results

Model           |BatchSize  |Precision |FPS       |Top-1(%)  |Top-5(%)
----------------|-----------|----------|----------|----------|--------
Squeezenet1_0   |    32     |   FP16   | 7777.50  |  58.08   | 80.39
