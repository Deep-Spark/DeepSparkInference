# Inception V3 (ixRT)

## Model Description

Inception v3 is a convolutional neural network architecture designed for image recognition and classification tasks. Developed by Google, it represents an evolution of the earlier Inception models. Inception v3 is characterized by its deep architecture, featuring multiple layers with various filter sizes and efficient use of computational resources. The network employs techniques like factorized convolutions and batch normalization to enhance training stability and accelerate convergence.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.09 |
| MR-V100 | 4.2.0 | 25.03 |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth>

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
python3 ../../ixrt_common/export.py --model-name inception_v3 --weight inception_v3_google-0cc3c7bd.pth --output checkpoints/inception_v3.onnx
```

## Model Inference

```bash
export PROJ_DIR=./
export DATASETS_DIR=/path/to/imagenet_val/
export CHECKPOINTS_DIR=./checkpoints
export RUN_DIR=../../ixrt_common/
export CONFIG_DIR=../../ixrt_common/config/INCEPTION_V3_CONFIG
```

### FP16

```bash
# Accuracy
bash scripts/infer_inception_v3_fp16_accuracy.sh
# Performance
bash scripts/infer_inception_v3_fp16_performance.sh
```

### INT8

```bash
# Accuracy
bash scripts/infer_inception_v3_int8_accuracy.sh
# Performance
bash scripts/infer_inception_v3_int8_performance.sh
```

## Model Results

| Model        | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
| :----: | :----: | :----: | :----: | :----: | :----: |
| Inception_v3 | 32        | FP16      | 3515.29 | 70.64    | 89.33    |
| Inception_v3 | 32        | INT8      | 4916.32 | 70.45    | 89.28    |
