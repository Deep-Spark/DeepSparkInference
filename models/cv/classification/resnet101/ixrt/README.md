# Resnet101 (IxRT)

## Model Description

ResNet-101 is a variant of the ResNet (Residual Network) architecture, and it belongs to a family of deep neural networks introduced by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun in their 2016 paper, "Deep Residual Learning for Image Recognition." The ResNet architecture is known for its effective use of residual connections, which help in training very deep neural networks.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.2.0     |  25.03  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/resnet101-63fe2227.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

pip3 install -r ../../ixrt_common/reuirements.txt
```

### Model Conversion

```bash
mkdir checkpoints
python3 ../../ixrt_common/export.py --model-name resnet101 --weight resnet101-63fe2227.pth --output checkpoints/resnet101.onnx
```

## Model Inference

```bash
export PROJ_DIR=./
export DATASETS_DIR=/path/to/imagenet_val/
export CHECKPOINTS_DIR=./checkpoints
export RUN_DIR=../../ixrt_common/
export CONFIG_DIR=../../ixrt_common/config/RESNET101_CONFIG
```

### FP16

```bash
# Accuracy
bash scripts/infer_resnet101_fp16_accuracy.sh
# Performance
bash scripts/infer_resnet101_fp16_performance.sh
```

### INT8

```bash
# Accuracy
bash scripts/infer_resnet101_int8_accuracy.sh
# Performance
bash scripts/infer_resnet101_int8_performance.sh
```

## Model Results

| Model     | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
| :----: | :----: | :----: | :----: | :----: | :----: |
| Resnet101 | 32        | FP16      | 2592.04 | 77.36    | 93.56    |
| Resnet101 | 32        | INT8      | 5760.69 | 76.88    | 93.43    |
