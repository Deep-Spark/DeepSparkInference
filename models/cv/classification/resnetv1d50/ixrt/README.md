# ResNetV1D50 (IxRT)

## Model Description

Residual Networks, or ResNets, learn residual functions with reference to the layer inputs, instead of learning unreferenced functions. Instead of hoping each few stacked layers directly fit a desired underlying mapping, residual nets let these layers fit a residual mapping.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.2.0     |  25.03  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.openmmlab.com/mmclassification/v0/resnet/resnetv1d50_b32x8_imagenet_20210531-db14775a.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

pip3 install -r ../../ixrt_common/requirments.txt
pip3 install mmcv==1.5.3 mmcls==0.24.0
```

### Model Conversion

```bash
# git clone mmpretrain
git clone -b v0.24.0 https://github.com/open-mmlab/mmpretrain.git

mkdir checkpoints

# export onnx model
python3 ../../ixrt_common/export_mmcls.py --cfg mmpretrain/configs/resnet/resnetv1d50_b32x8_imagenet.py --weight resnetv1d50_b32x8_imagenet_20210531-db14775a.pth --output checkpoints/resnet_v1_d50.onnx
```

## Model Inference

```bash
export PROJ_DIR=./
export DATASETS_DIR=/path/to/imagenet_val/
export CHECKPOINTS_DIR=./checkpoints
export RUN_DIR=../../ixrt_common/
export CONFIG_DIR=../../ixrt_common/config/RESNETV1D50_CONFIG
```

### FP16

```bash
# Accuracy
bash scripts/infer_resnetv1d50_fp16_accuracy.sh
# Performance
bash scripts/infer_resnetv1d50_fp16_performance.sh
```

### INT8

```bash
# Accuracy
bash scripts/infer_resnetv1d50_int8_accuracy.sh
# Performance
bash scripts/infer_resnetv1d50_int8_performance.sh
```

## Model Results

| Model         | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
| ------------- | --------- | --------- | ------- | -------- | -------- |
| ResNet_V1_D50 | 32        | FP16      | 3887.55 | 0.77544  | 0.93568  |
| ResNet_V1_D50 | 32        | INT8      | 7148.58 | 0.7711   | 0.93514  |
