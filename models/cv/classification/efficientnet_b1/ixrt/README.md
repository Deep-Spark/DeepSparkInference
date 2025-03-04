# EfficientNet B1

## Model Description

EfficientNet B1 is one of the variants in the EfficientNet family of neural network architectures, introduced by Mingxing Tan and Quoc V. Le in their paper "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." EfficientNet B1 is a scaled-up version of the baseline model (B0) and is designed to achieve better performance on various computer vision tasks.

## Model Preparation

### Install Dependencies

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

pip3 install -r requirements.txt
```

### Prepare Resources

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Model Conversion

```bash
mkdir checkpoints
python3 export_onnx.py --output_model checkpoints/efficientnet-b1.onnx
```

## Model Inference

```bash
export PROJ_DIR=./
export DATASETS_DIR=/path/to/imagenet_val/
export CHECKPOINTS_DIR=./checkpoints
export RUN_DIR=./
export CONFIG_DIR=config/EFFICIENTNET_B1_CONFIG
```

### FP16

```bash
# Accuracy
bash scripts/infer_efficientnet_b1_fp16_accuracy.sh
# Performance
bash scripts/infer_efficientnet_b1_fp16_performance.sh
```

### INT8

```bash
# Accuracy
bash scripts/infer_efficientnet_b1_int8_accuracy.sh
# Performance
bash scripts/infer_efficientnet_b1_int8_performance.sh
```

## Model Results

Model           |BatchSize  |Precision |FPS      |Top-1(%)  |Top-5(%)
----------------|-----------|----------|---------|----------|--------
EfficientNet_B1 |    32     |   FP16   | 1517.84 |  77.60   | 93.60
EfficientNet_B1 |    32     |   INT8   | 1817.88 |  75.32   | 92.46
