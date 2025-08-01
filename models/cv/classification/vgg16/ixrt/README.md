# VGG16 (ixRT)

## Model Description

VGG16 is a deep convolutional neural network model developed by the Visual Geometry Group at the University of Oxford.
It finished second in the 2014 ImageNet Massive Visual Identity Challenge (ILSVRC).

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.09 |
| MR-V100 | 4.2.0 | 25.03 |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/vgg16-397923af.pth>

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
python3 ../../ixrt_common/export.py --model-name vgg16 --weight vgg16-397923af.pth --output checkpoints/vgg16.onnx
```

## Model Inference

```bash
export PROJ_DIR=./
export DATASETS_DIR=/path/to/imagenet_val/
export CHECKPOINTS_DIR=./checkpoints
export RUN_DIR=../../ixrt_common/
export CONFIG_DIR=../../ixrt_common/config/VGG16_CONFIG
```

### FP16

```bash
# Accuracy
bash scripts/infer_vgg16_fp16_accuracy.sh
# Performance
bash scripts/infer_vgg16_fp16_performance.sh
```

### INT8

```bash
# Accuracy
bash scripts/infer_vgg16_int8_accuracy.sh
# Performance
bash scripts/infer_vgg16_int8_performance.sh
```

## Model Results

| Model | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
| :----: | :----: | :----: | :----: | :----: | :----: |
| VGG16 | 32        | FP16      | 1777.85 | 71.57    | 90.40    |
| VGG16 | 32        | INT8      | 4451.80 | 71.47    | 90.35    |
