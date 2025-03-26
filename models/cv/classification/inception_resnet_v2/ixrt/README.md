# Inception-ResNet-V2 (IxRT)

## Model Description

Inception-ResNet-V2 is a deep learning model proposed by Google in 2016, which combines the architectures of Inception and ResNet. This model integrates the dense connections of the Inception series with the residual connections of ResNet, aiming to enhance model performance and training efficiency.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| MR-V100 | 4.2.0     |  25.03  |

## Model Preparation

### Prepare Resources

Pretrained model: <http://data.lip6.fr/cadene/pretrainedmodels/inceptionresnetv2-520b38e4.pth>

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
python3 export_model.py --output_model /Path/to/checkpoints/inceptionresnetv2.onnx
```

## Model Inference

```bash
export PROJ_DIR=/Path/to/inceptionresnetv2/ixrt
export DATASETS_DIR=/path/to/imagenet_val/
export CHECKPOINTS_DIR=./checkpoints
export RUN_DIR=/Path/to/inceptionresnetv2/ixrt
export CONFIG_DIR=/Path/to/config/INCEPTIONRESNETV2_CONFIG
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```

### FP16

```bash
# Accuracy
bash scripts/infer_inceptionresnetv2_fp16_accuracy.sh
# Performance
bash scripts/infer_inceptionresnetv2_fp16_performance.sh
```

### INT8

```bash
# Accuracy
bash scripts/infer_inceptionresnetv2_int8_accuracy.sh
# Performance
bash scripts/infer_inceptionresnetv2_int8_performance.sh
```

## Model Results

| Model               | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
|---------------------|-----------|-----------|---------|----------|----------|
| Inception-ResNet-V2 | 64        | FP16      | 871.74  | 80.20    | 95.18    |
| Inception-ResNet-V2 | 64        | INT8      | 1059.35 | 79.73    | 95.04    |
