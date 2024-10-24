# ShuffleNetV2_x0_5

## Description

ShuffleNetV2_x0_5 is a lightweight convolutional neural network architecture designed for efficient image classification and feature extraction, it also incorporates other design optimizations such as depthwise separable convolutions, group convolutions, and efficient building blocks to further reduce computational complexity and improve efficiency.

## Setup

### Install

```bash
pip3 install -r requirements.txt
```

### Download

Pretrained model: <https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Model Conversion

```bash
python3 export.py --weight shufflenetv2_x0.5-f707e7126e.pth --output shufflenetv2_x0_5.onnx
```

## Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
```

### FP16

```bash
# Accuracy
bash scripts/infer_shufflenetv2_x0_5_fp16_accuracy.sh
# Performance
bash scripts/infer_shufflenetv2_x0_5_fp16_performance.sh
```

## Results

Model             |BatchSize  |Precision |FPS       |Top-1(%)  |Top-5(%)
------------------|-----------|----------|----------|----------|--------
ShuffleNetV2_x0_5 |    32     |   FP16   | 11677.55 |  60.501  |  81.702
