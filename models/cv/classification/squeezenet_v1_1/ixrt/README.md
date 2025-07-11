# SqueezeNet 1.1 (IxRT)

## Model Description

SqueezeNet 1.1 is a deep learning model for image classification, designed to be lightweight and efficient for deployment on resource-constrained devices.

It was developed by researchers at DeepScale and released in 2016.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.09 |
| MR-V100 | 4.2.0 | 25.03 |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/squeezenet1_1-b8a52dc0.pth>

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
python3 ../../ixrt_common/export.py --model-name squeezenet1_1 --weight squeezenet1_1-b8a52dc0.pth --output checkpoints/squeezenet_v1_1.onnx
```

## Model Inference

```bash
export PROJ_DIR=./
export DATASETS_DIR=/path/to/imagenet_val/
export CHECKPOINTS_DIR=./checkpoints
export RUN_DIR=../../ixrt_common/
export CONFIG_DIR=../../ixrt_common/config/SQUEEZENET_V1_1_CONFIG

```

### FP16

```bash
# Accuracy
bash scripts/infer_squeezenet_v1_1_fp16_accuracy.sh
# Performance
bash scripts/infer_squeezenet_v1_1_fp16_performance.sh
```

### INT8

```bash
# Accuracy
bash scripts/infer_squeezenet_v1_1_int8_accuracy.sh
# Performance
bash scripts/infer_squeezenet_v1_1_int8_performance.sh
```

## Model Results

| Model          | BatchSize | Precision | FPS   | Top-1(%) | Top-5(%) |
| -------------- | --------- | --------- | ----- | -------- | -------- |
| SqueezeNet 1.1 | 32        | FP16      | 13701 | 0.58182  | 0.80622  |
| SqueezeNet 1.1 | 32        | INT8      | 20128 | 0.50966  | 0.77552  |
