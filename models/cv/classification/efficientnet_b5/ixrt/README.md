# EfficientNet B5 (ixRT)

## Model Description

EfficientNet B5 is a member of the EfficientNet family, a series of convolutional neural network architectures that are designed to achieve excellent accuracy and efficiency. Introduced by researchers at Google, EfficientNets utilize the compound scaling method, which uniformly scales the depth, width, and resolution of the network to improve accuracy and efficiency.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0     |  25.09  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/efficientnet_b5_lukemelas-1a07897c.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
pip3 install -r ../../ixrt_common/requirements.txt
```

### Model Conversion

```bash
mkdir checkpoints
python3 ../../ixrt_common/export.py --model-name efficientnet_b5 --weight efficientnet_b5_lukemelas-1a07897c.pth --output checkpoints/efficientnet_b5.onnx
```

## Model Inference

```bash
export PROJ_DIR=./
export DATASETS_DIR=/path/to/imagenet_val/
export CHECKPOINTS_DIR=./checkpoints
export RUN_DIR=../../ixrt_common/
export CONFIG_DIR=../../ixrt_common/config/EFFICIENTNET_B5_CONFIG
```

### FP16

```bash
# Accuracy
bash scripts/infer_efficientnet_b5_fp16_accuracy.sh
# Performance
bash scripts/infer_efficientnet_b5_fp16_performance.sh
```

## Model Results

| Model           | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
| --------------- | --------- | --------- | ------- | -------- | -------- |
| EfficientNet_B5 | 32        | FP16      | 879.44 | 73.15    | 90.94    |
