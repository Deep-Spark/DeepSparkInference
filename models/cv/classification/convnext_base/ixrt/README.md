# ConvNeXt Base (IxRT)

## Model Description

The ConvNeXt Base model represents a significant stride in the evolution of convolutional neural networks (CNNs), introduced by researchers at Facebook AI Research (FAIR) and UC Berkeley. It is part of the ConvNeXt family, which challenges the dominance of Vision Transformers (ViTs) in the realm of visual recognition tasks.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.09 |
| MR-V100 | 4.2.0 | 25.03 |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/convnext_base-6075fbad.pth>

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
python3 ../../ixrt_common/export.py --model-name convnext_base --weight convnext_base-6075fbad.pth --output checkpoints/convnext_base.onnx
```

## Model Inference

```bash
export PROJ_DIR=./
export DATASETS_DIR=/path/to/imagenet_val/
export CHECKPOINTS_DIR=./checkpoints
export RUN_DIR=../../ixrt_common/
export CONFIG_DIR=../../ixrt_common/config/CONVNEXT_BASE_CONFIG
```

### FP16

```bash

# Accuracy
bash scripts/infer_convnext_base_fp16_accuracy.sh
# Performance
bash scripts/infer_convnext_base_fp16_performance.sh
```

## Model Results

| Model          | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
| -------------- | --------- | --------- | ------- | -------- | -------- |
| ConvNext Base  | 32        | FP16      | 219.817 | 83.70    | 96.73    |
