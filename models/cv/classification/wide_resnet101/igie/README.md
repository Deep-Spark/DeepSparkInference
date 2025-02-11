# Wide ResNet101

## Description

Wide ResNet101 is a variant of the ResNet architecture that focuses on increasing the network's width (number of channels per layer) rather than its depth. This approach, inspired by the paper "Wide Residual Networks," balances model depth and width to achieve better performance while avoiding the drawbacks of overly deep networks, such as vanishing gradients and feature redundancy.Wide ResNet101 builds upon the standard ResNet101 architecture but doubles (or quadruples) the number of channels in each residual block. This results in significantly improved feature representation, making it suitable for complex tasks like image classification, object detection, and segmentation.

## Setup

### Install

```bash
pip3 install -r requirements.txt
```

### Download

Pretrained model: <https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Model Conversion

```bash
python3 export.py --weight wide_resnet101_2-32ee1156.pth --output wide_resnet101.onnx
```

## Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
```

### FP16

```bash
# Accuracy
bash scripts/infer_wide_resnet101_fp16_accuracy.sh
# Performance
bash scripts/infer_wide_resnet101_fp16_performance.sh
```

## Results

| Model          | BatchSize | Precision | FPS      | Top-1(%) | Top-5(%) |
| -------------- | --------- | --------- | -------- | -------- | -------- |
| Wide ResNet101 | 32        | FP16      | 1339.037 | 78.459   | 94.052   |
