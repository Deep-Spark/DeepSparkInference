# AlexNet (IxRT)

## Model Description

AlexNet is a classic convolutional neural network architecture. It consists of convolutions, max pooling and dense
layers as the basic building blocks.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| MR-V100 | 4.2.0     |  25.03  |

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

pip3 install -r requirements.txt
```

### Model Conversion

```bash
mkdir checkpoints
python3 export_onnx.py --origin_model /path/to/alexnet-owt-7be5be79.pth --output_model checkpoints/alexnet.onnx
```

## Model Inference

```bash
export PROJ_DIR=./
export DATASETS_DIR=/path/to/imagenet_val/
export CHECKPOINTS_DIR=./checkpoints
export RUN_DIR=./
export CONFIG_DIR=config/ALEXNET_CONFIG
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
|---------|-----------|-----------|----------|----------|----------|
| AlexNet | 32        | FP16      | 17644.90 | 56.54    | 79.08    |
| AlexNet | 32        | INT8      | 18276.83 | 55.37    | 79.04    |
