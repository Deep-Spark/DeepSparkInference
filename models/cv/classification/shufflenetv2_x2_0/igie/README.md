# ShuffleNetV2_x2_0

## Description

ShuffleNetV2_x2_0 is a lightweight convolutional neural network introduced in the paper "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design" by Megvii (Face++). It is designed to achieve high performance with low computational cost, making it ideal for mobile and embedded devices.The x2_0 in its name indicates a width multiplier of 2.0, meaning the model has twice as many channels compared to the baseline ShuffleNetV2_x1_0. It employs Channel Shuffle to enable efficient information exchange between grouped convolutions, addressing the limitations of group convolutions. The core building block, the ShuffleNetV2 block, features a split-merge design and channel shuffle mechanism, ensuring both high efficiency and accuracy.

## Setup

### Install

```bash
pip3 install -r requirements.txt
```

### Download

Pretrained model: <https://download.pytorch.org/models/shufflenetv2_x2_0-8be3c8ee.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Model Conversion

```bash
python3 export.py --weight shufflenetv2_x2_0-8be3c8ee.pth --output shufflenetv2_x2_0.onnx
```

## Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
```

### FP16

```bash
# Accuracy
bash scripts/infer_shufflenetv2_x2_0_fp16_accuracy.sh
# Performance
bash scripts/infer_shufflenetv2_x2_0_fp16_performance.sh
```

## Results

| Model             | BatchSize | Precision | FPS      | Top-1(%) | Top-5(%) |
| ----------------- | --------- | --------- | -------- | -------- | -------- |
| ShuffleNetV2_x2_0 | 32        | FP16      | 5439.098 | 76.176   | 92.860   |