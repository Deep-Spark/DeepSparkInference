# EfficientNet B1 (IxRT)

## Model Description

EfficientNet B1 is one of the variants in the EfficientNet family of neural network architectures, introduced by Mingxing Tan and Quoc V. Le in their paper "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." EfficientNet B1 is a scaled-up version of the baseline model (B0) and is designed to achieve better performance on various computer vision tasks.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.2.0     |  25.03  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/efficientnet_b1-c27df63c.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

pip3 install -r ../../ixrt_common/requirements.txt
```

### Model Conversion

```bash
mkdir checkpoints
python3 ../../ixrt_common/export.py --model-name efficientnet_b1 --weight efficientnet_b1-c27df63c.pth --output checkpoints/efficientnet_b1.onnx
```

## Model Inference

```bash
export PROJ_DIR=./
export DATASETS_DIR=/path/to/imagenet_val/
export CHECKPOINTS_DIR=./checkpoints
export RUN_DIR=../../ixrt_common/
export CONFIG_DIR=../../ixrt_common/config/EFFICIENTNET_B1_CONFIG
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

| Model           | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
| :----: | :----: | :----: | :----: | :----: | :----: |
| EfficientNet_B1 | 32        | FP16      | 1517.84 | 77.60    | 93.60    |
| EfficientNet_B1 | 32        | INT8      | 1817.88 | 75.32    | 92.46    |
