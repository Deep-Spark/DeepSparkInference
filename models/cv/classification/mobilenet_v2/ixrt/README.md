# MobileNetV2

## Description

The MobileNetV2 architecture is based on an inverted residual structure where the input and output of the residual block are thin bottleneck layers opposite to traditional residual models which use expanded representations in the input an MobileNetV2 uses lightweight depthwise convolutions to filter features in the intermediate expansion layer.

## Setup

### Install

```bash
pip3 install -r requirements.txt
```

### Download

Pretrained model: <https://download.pytorch.org/models/mobilenet_v2-b0353104.pth>

Download the [imagenet](https://www.image-net.org/download.php) to download the validation dataset.

### Model Conversion

```bash
mkdir checkpoints
python3 python/export_onnx.py --origin_model /path/to/mobilenet_v2-b0353104 --output_model checkpoints/mobilenet_v2.onnx
```

## Inference

```bash
export PROJ_DIR=./
export DATASETS_DIR=/path/to/imagenet_val/
export CHECKPOINTS_DIR=./checkpoints
export RUN_DIR=./
```

### FP16

```bash
# Test ACC
bash script/infer_mobilenetv2_fp16_accuracy.sh
# Test FPS
bash script/infer_mobilenetv2_fp16_performance.sh
```

### INT8

```bash
# Test ACC
bash script/infer_mobilenetv2_int8_accuracy.sh
# Test FPS
bash script/infer_mobilenetv2_int8_performance.sh
```

## Results

| Model       | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
| ----------- | --------- | --------- | ------- | -------- | -------- |
| MobileNetV2 | 32        | FP16      | 4835.19 | 0.7186   | 0.90316  |

## Referenece

- [MobileNetV2](https://arxiv.org/abs/1801.04381)
- 
