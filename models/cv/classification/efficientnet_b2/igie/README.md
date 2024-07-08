# EfficientNet B2

## Description

EfficientNet B2 is a member of the EfficientNet family, a series of convolutional neural network architectures that are designed to achieve excellent accuracy and efficiency. Introduced by researchers at Google, EfficientNets utilize the compound scaling method, which uniformly scales the depth, width, and resolution of the network to improve accuracy and efficiency.

## Setup

### Install

```bash
pip3 install onnx
pip3 install tqdm
```

### Download

Pretrained model: <https://download.pytorch.org/models/efficientnet_b2_rwightman-c35c1473.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Model Conversion

```bash
python3 export.py --weight efficientnet_b2_rwightman-c35c1473.pth --output efficientnet_b2.onnx
```

## Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
```

### FP16

```bash
# Accuracy
bash scripts/infer_efficientnet_b2_fp16_accuracy.sh
# Performance
bash scripts/infer_efficientnet_b2_fp16_performance.sh
```

## Results

Model           |BatchSize  |Precision |FPS      |Top-1(%) |Top-5(%)
----------------|-----------|----------|---------|---------|--------
Efficientnet_b2 |    32     |   FP16   | 1010.90 | 77.75   | 93.70
