# EfficientNet B0 (IxRT)

## Model Description

EfficientNet B0 is a convolutional neural network architecture that belongs to the EfficientNet family, which was introduced by Mingxing Tan and Quoc V. Le in their paper "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." The EfficientNet family is known for achieving state-of-the-art performance on various computer vision tasks while being more computationally efficient than many existing models.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.2.0     |  25.03  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth>

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
python3 ../../ixrt_common/export.py --model-name efficientnet_b0 --weight /path/to/efficientnet_b0_rwightman-3dd342df.pth --output checkpoints/efficientnet_b0.onnx
```

## Model Inference

```bash
export PROJ_DIR=./
export DATASETS_DIR=/path/to/imagenet_val/
export CHECKPOINTS_DIR=./checkpoints
export RUN_DIR=../../ixrt_common/
export CONFIG_DIR=../../ixrt_common/config/EFFICIENTNET_B0_CONFIG
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

## Model Results

| Model           | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
| :----: | :----: | :----: | :----: | :----: | :----: |
| EfficientNet B0 | 32        | FP16      | 2325.54 | 77.66    | 93.58    |
| EfficientNet B0 | 32        | INT8      | 2666.00 | 74.27    | 91.85    |
