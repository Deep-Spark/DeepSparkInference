# EfficientNet B0

## Description

EfficientNet B0 is a convolutional neural network architecture that belongs to the EfficientNet family, which was introduced by Mingxing Tan and Quoc V. Le in their paper "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." The EfficientNet family is known for achieving state-of-the-art performance on various computer vision tasks while being more computationally efficient than many existing models.

## Setup

### Install

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

pip3 install -r requirements.txt
```

### Download

Pretrained model: <https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Model Conversion

```bash
python3 export_onnx.py --origin_model /path/to/efficientnet_b0_rwightman-3dd342df.pth --output_model efficientnet_b0.onnx
```

## Inference

```bash
export DATASETS_DIR=/path/to/imagenet_val/
```

### FP16

```bash
# Accuracy
bash scripts/infer_efficientnet_b0_fp16_accuracy.sh
# Performance
bash scripts/infer_efficientnet_b0_fp16_performance.sh
```

### INT8

```bash
# Accuracy
bash scripts/infer_efficientnet_b0_int8_accuracy.sh
# Performance
bash scripts/infer_efficientnet_b0_int8_performance.sh
```

## Results

| Model           | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
| --------------- | --------- | --------- | ------- | -------- | -------- |
| EfficientNet B0 | 32        | FP16      | 2325.54 | 77.66    | 93.58    |
| EfficientNet B0 | 32        | INT8      | 2666.00 | 74.27    | 91.85    |
