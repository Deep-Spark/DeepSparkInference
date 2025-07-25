# ResNet50 (ixRT)

## Model Description

Residual Networks, or ResNets, learn residual functions with reference to the layer inputs, instead of learning unreferenced functions. Instead of hoping each few stacked layers directly fit a desired underlying mapping, residual nets let these layers fit a residual mapping.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.09 |
| MR-V100 | 4.2.0 | 25.03 |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/resnet50-0676ba61.pth>

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
python3 ../../ixrt_common/export.py --model-name resnet50 --weight resnet50-0676ba61.pth --output checkpoints/resnet50.onnx
```

## Model Inference

```bash
export PROJ_DIR=./
export DATASETS_DIR=/path/to/imagenet_val/
export CHECKPOINTS_DIR=./checkpoints
export RUN_DIR=../../ixrt_common/
export CONFIG_DIR=../../ixrt_common/config/RESNET50_CONFIG
```

### FP16

```bash
# Accuracy
bash scripts/infer_resnet50_fp16_accuracy.sh
# Performance
bash scripts/infer_resnet50_fp16_performance.sh
```

### INT8

```bash
# Accuracy
bash scripts/infer_resnet50_int8_accuracy.sh
# Performance
bash scripts/infer_resnet50_int8_performance.sh
```

## Model Results

| Model    | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
| :----: | :----: | :----: | :----: | :----: | :----: |
| ResNet50 | 32        | FP16      | 4077.58 | 0.76158  | 0.92872  |
| ResNet50 | 32        | INT8      | 9113.07 | 0.74516  | 0.9287   |
