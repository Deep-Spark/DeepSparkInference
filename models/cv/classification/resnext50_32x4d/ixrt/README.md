# ResNext50_32x4d (ixRT)

## Model Description

The ResNeXt50_32x4d model is a convolutional neural network architecture designed for image classification tasks. It is an extension of the ResNet (Residual Network) architecture, incorporating the concept of cardinality to enhance model performance.

## Supported Environments

| GPU | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release | Branch |
| :----: | :----: | :----: | :----: |
| MR-V100 | 4.4.0 | 26.03 | release/26.03 |
| MR-V100 | 4.3.0 | 25.12 | release/25.12 |

> **Note:** 请切换到与您的 SDK 版本对应的 Release 分支进行测试。请勿直接在 master 分支上运行测试，因为 master 分支可能包含与您的本地 SDK 版本不兼容的最新更改。
>
> 切换分支命令示例：`git checkout release/26.03`

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth>

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
python3 ../../ixrt_common/export.py --model-name resnext50_32x4d --weight resnext50_32x4d-7cdf4587.pth --output checkpoints/resnext50_32x4d.onnx
```

## Model Inference

```bash
export PROJ_DIR=./
export DATASETS_DIR=/path/to/imagenet_val/
export CHECKPOINTS_DIR=./checkpoints
export RUN_DIR=../../ixrt_common/
export CONFIG_DIR=../../ixrt_common/config/RESNEXT50_32X4D_CONFIG
```

### FP16

```bash
# Accuracy
bash scripts/infer_resnext50_32x4d_fp16_accuracy.sh
# Performance
bash scripts/infer_resnext50_32x4d_fp16_performance.sh
```

## Model Results

| Model           | BatchSize | Precision | FPS    | Top-1(%) | Top-5(%) |
| --------------- | --------- | --------- | ------ | -------- | -------- |
| resnext50_32x4d | 32        | FP16      | 417.01 | 77.614   | 93.686   |
