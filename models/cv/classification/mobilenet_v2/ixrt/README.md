# MobileNetV2

## Description

The MobileNetV2 architecture is based on an inverted residual structure where the input and output of the residual block are thin bottleneck layers opposite to traditional residual models which use expanded representations in the input an MobileNetV2 uses lightweight depthwise convolutions to filter features in the intermediate expansion layer.

## Setup

### Install

```bash
pip3 install tqdm
pip3 install onnxsim
pip3 install opencv-python
pip3 install ppq
pip3 install protobuf==3.20.0
```

### Download

Download the [imagenet](https://www.image-net.org/download.php) validation dataset, and place in `${PROJ_ROOT}/data/datasets`;

## Inference

### FP16

```bash
cd python/
# Test ACC
bash script/infer_mobilenetv2_fp16_accuary.sh
# Test FPS
bash script/infer_mobilenetv2_fp16_performance.sh
```

### INT8

```bash
# Test ACC
bash script/infer_mobilenetv2_int8_accuary.sh
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