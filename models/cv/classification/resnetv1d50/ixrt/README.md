# ResNetV1D50

## Model Description

Residual Networks, or ResNets, learn residual functions with reference to the layer inputs, instead of learning unreferenced functions. Instead of hoping each few stacked layers directly fit a desired underlying mapping, residual nets let these layers fit a residual mapping.

## Model Preparation

### Install Dependencies

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

pip3 install -r requirments.txt
```

### Prepare Resources

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Model Conversion

```bash
mkdir checkpoints
python3 export_onnx.py --output_model checkpoints/resnet_v1_d50.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/path/to/imagenet_val/
export CHECKPOINTS_DIR=./checkpoints
export RUN_DIR=./
export CONFIG_DIR=config/RESNETV1D50_CONFIG
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
