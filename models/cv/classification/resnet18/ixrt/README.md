# ResNet18 (ixRT)

## Model Description

ResNet-18 is a variant of the ResNet (Residual Network) architecture, which was introduced by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun in their 2016 paper, "Deep Residual Learning for Image Recognition." The ResNet architecture was pivotal in addressing the challenges of training very deep neural networks by introducing residual blocks.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.09 |
| MR-V100 | 4.2.0 | 25.03 |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/resnet18-f37072fd.pth>

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
python3 ../../ixrt_common/export.py --model-name resnet18 --weight resnet18-f37072fd.pth --output checkpoints/resnet18.onnx
```

## Model Inference

```bash
export PROJ_DIR=./
export DATASETS_DIR=/path/to/imagenet_val/
export CHECKPOINTS_DIR=./checkpoints
export RUN_DIR=../../ixrt_common/
export CONFIG_DIR=../../ixrt_common/config/RESNET18_CONFIG
```

### FP16

```bash
# Accuracy
bash scripts/infer_resnet18_fp16_accuracy.sh
# Performance
bash scripts/infer_resnet18_fp16_performance.sh
```

### INT8

```bash
# Accuracy
bash scripts/infer_resnet18_int8_accuracy.sh
# Performance
bash scripts/infer_resnet18_int8_performance.sh
```

## Model Results

| Model    | BatchSize | Precision | FPS      | Top-1(%) | Top-5(%) |
| :----: | :----: | :----: | :----: | :----: | :----: |
| Resnet18 | 32        | FP16      | 9592.98  | 69.77    | 89.09    |
| Resnet18 | 32        | INT8      | 21314.55 | 69.53    | 88.97    |
