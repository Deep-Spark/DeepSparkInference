# EfficientNetV2 (IxRT)

## Model Description

EfficientNetV2 is an improved version of the EfficientNet architecture proposed by Google, aiming to enhance model
performance and efficiency. Unlike the original EfficientNet, EfficientNetV2 features a simplified design and
incorporates a series of enhancement strategies to further boost performance.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.2.0     |  25.03  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnetv2_t_agc-3620981a.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

pip3 install -r ../../ixrt_common/requirements.txt
pip3 install timm==1.0.11
```

### Model Conversion

```bash
mkdir checkpoints
git clone -b v1.0.11 --depth=1 https://github.com/huggingface/pytorch-image-models.git
cp ./export_onnx.py pytorch-image-models/timm/models
cp ./_builder.py pytorch-image-models/timm/models
cd pytorch-image-models/timm
mkdir -p /root/.cache/torch/hub/checkpoints/
wget -P /root/.cache/torch/hub/checkpoints/ https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnetv2_t_agc-3620981a.pth
python3 -m models.export_onnx --output_model ../../checkpoints/efficientnet_v2.onnx
cd ../../
```

## Model Inference

```bash
export PROJ_DIR=./
export DATASETS_DIR=/path/to/imagenet_val/
export CHECKPOINTS_DIR=./checkpoints
export RUN_DIR=../../ixrt_common/
export CONFIG_DIR=../../ixrt_common/config/EFFICIENTNET_V2_CONFIG
```

### FP16

```bash
# Accuracy
bash scripts/infer_efficientnet_v2_fp16_accuracy.sh
# Performance
bash scripts/infer_efficientnet_v2_fp16_performance.sh
```

### INT8

```bash
# Accuracy
bash scripts/infer_efficientnet_v2_int8_accuracy.sh
# Performance
bash scripts/infer_efficientnet_v2_int8_performance.sh
```

## Model Results

| Model          | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
|----------------|-----------|-----------|---------|----------|----------|
| EfficientnetV2 | 32        | FP16      | 1882.87 | 82.14    | 96.16    |
| EfficientnetV2 | 32        | INT8      | 2595.96 | 81.50    | 95.96    |
