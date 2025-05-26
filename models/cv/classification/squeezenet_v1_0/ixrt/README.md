# SqueezeNet 1.0 (IxRT)

## Model Description

SqueezeNet 1.0 is a deep learning model for image classification, designed to be lightweight and efficient for deployment on resource-constrained devices.

It was developed by researchers at DeepScale and released in 2016.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.2.0     |  25.03  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/squeezenet1_0-b66bff10.pth>

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
python3 ../../ixrt_common/export.py --model-name squeezenet1_0 --weight squeezenet1_0-b66bff10.pth --output checkpoints/squeezenetv10.onnx
```

## Model Inference

```bash
export PROJ_DIR=./
export DATASETS_DIR=/path/to/imagenet_val/
export CHECKPOINTS_DIR=./checkpoints
export RUN_DIR=../../ixrt_common/
export CONFIG_DIR=../../ixrt_common/config/SQUEEZENET_V1_0_CONFIG
```

### FP16

```bash
# Accuracy
bash scripts/infer_squeezenet_v1_0_fp16_accuracy.sh
# Performance
bash scripts/infer_squeezenet_v1_0_fp16_performance.sh
```

### INT8

```bash
# Accuracy
bash scripts/infer_squeezenet_v1_0_int8_accuracy.sh
# Performance
bash scripts/infer_squeezenet_v1_0_int8_performance.sh
```

## Model Results

| Model          | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
| :----: | :----: | :----: | :----: | :----: | :----: |
| SqueezeNet 1.0 | 32        | FP16      | 7740.26 | 58.07    | 80.43    |
| SqueezeNet 1.0 | 32        | INT8      | 8871.93 | 55.10    | 79.21    |
