# GoogLeNet (IxRT)

## Model Description

GoogLeNet is a type of convolutional neural network based on the Inception architecture. It utilises Inception modules, which allow the network to choose between multiple convolutional filter sizes in each block. An Inception network stacks these modules on top of each other, with occasional max-pooling layers with stride 2 to halve the resolution of the grid.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.09 |
| MR-V100 | 4.2.0 | 25.03 |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/googlenet-1378be20.pth>

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
python3 ../../ixrt_common/export.py --model-name googlenet --weight googlenet-1378be20.pth --output checkpoints/googlenet.onnx
```

## Model Inference

```bash
export PROJ_DIR=./
export DATASETS_DIR=/path/to/imagenet_val/
export CHECKPOINTS_DIR=./checkpoints
export RUN_DIR=../../ixrt_common/
export CONFIG_DIR=../../ixrt_common/config/GOOGLENET_CONFIG
```

### FP16

```bash
# Accuracy
bash scripts/infer_googlenet_fp16_accuracy.sh
# Performance
bash scripts/infer_googlenet_fp16_performance.sh
```

### INT8

```bash
# Accuracy
bash scripts/infer_googlenet_int8_accuracy.sh
# Performance
bash scripts/infer_googlenet_int8_performance.sh
```

## Model Results

| Model     | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
| :----: | :----: | :----: | :----: | :----: | :----: |
| GoogLeNet | 32        | FP16      | 6470.34 | 62.456   | 84.33    |
| GoogLeNet | 32        | INT8      | 9358.11 | 62.106   | 84.30    |
