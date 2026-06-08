# AlexNet (ixRT)

## Model Description

AlexNet is a classic convolutional neural network architecture. It consists of convolutions, max pooling and dense
layers as the basic building blocks.

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

Pretrained model: <https://download.pytorch.org/models/alexnet-owt-7be5be79.pth>

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
python3 ../../ixrt_common/export.py --model-name alexnet --weight alexnet-owt-7be5be79.pth --output checkpoints/alexnet.onnx
```

## Model Inference

```bash
export PROJ_DIR=./
export DATASETS_DIR=/path/to/imagenet_val/
export CHECKPOINTS_DIR=./checkpoints
export RUN_DIR=../../ixrt_common/
export CONFIG_DIR=../../ixrt_common/config/ALEXNET_CONFIG
```

### FP16

```bash
# Accuracy
bash scripts/infer_alexnet_fp16_accuracy.sh
# Performance
bash scripts/infer_alexnet_fp16_performance.sh
```

### INT8

```bash
# Accuracy
bash scripts/infer_alexnet_int8_accuracy.sh
# Performance
bash scripts/infer_alexnet_int8_performance.sh
```

## Model Results

| Model   | BatchSize | Precision | FPS      | Top-1(%) | Top-5(%) |
| :----: | :----: | :----: | :----: | :----: | :----: |
| AlexNet | 32        | FP16      | 17644.90 | 56.54    | 79.08    |
| AlexNet | 32        | INT8      | 18276.83 | 55.37    | 79.04    |
