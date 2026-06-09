# ShuffleNetV2_x0_5 (ixRT)

## Model Description

ShuffleNetV2_x0_5 is a lightweight convolutional neural network architecture designed for efficient image classification
and feature extraction, it also incorporates other design optimizations such as depthwise separable convolutions, group
convolutions, and efficient building blocks to further reduce computational complexity and improve efficiency.

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

Pretrained model: <https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
pip3 install -r ../../ixrt_common/requirements.txt
```

### Model Conversion

```bash
mkdir checkpoints
python3 ../../ixrt_common/export.py --model-name shufflenet_v2_x0_5 --weight shufflenetv2_x0.5-f707e7126e.pth --output checkpoints/shufflenetv2_x0_5.onnx
```

## Model Inference

```bash
export PROJ_DIR=./
export DATASETS_DIR=/path/to/imagenet_val/
export CHECKPOINTS_DIR=./checkpoints
export RUN_DIR=../../ixrt_common/
export CONFIG_DIR=../../ixrt_common/config/SHUFFLENETV2_X0_5_CONFIG
```

### FP16

```bash
# Accuracy
bash scripts/infer_shufflenetv2_x0_5_fp16_accuracy.sh
# Performance
bash scripts/infer_shufflenetv2_x0_5_fp16_performance.sh
```

## Model Results

| Model             | BatchSize | Precision | FPS      | Top-1(%) | Top-5(%) |
| :----: | :----: | :----: | :----: | :----: | :----: |
| ShuffleNetV2 x0_5 | 32        | FP16      | 10680.65 | 60.53    | 81.74    |
