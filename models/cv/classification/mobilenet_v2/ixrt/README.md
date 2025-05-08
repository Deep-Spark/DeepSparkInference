# MobileNetV2 (IxRT)

## Model Description

The MobileNetV2 architecture is based on an inverted residual structure where the input and output of the residual block are thin bottleneck layers opposite to traditional residual models which use expanded representations in the input an MobileNetV2 uses lightweight depthwise convolutions to filter features in the intermediate expansion layer.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| MR-V100 | 4.2.0     |  25.03  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/mobilenet_v2-b0353104.pth>

Download the [imagenet](https://www.image-net.org/download.php) to download the validation dataset.

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
python3 ../../ixrt_common/export.py --model-name mobilenet_v2 --weight mobilenet_v2-b0353104.pth --output checkpoints/mobilenet_v2.onnx
```

## Model Inference

```bash
export PROJ_DIR=./
export DATASETS_DIR=/path/to/imagenet_val/
export CHECKPOINTS_DIR=./checkpoints
export RUN_DIR=../../ixrt_common/
export CONFIG_DIR=../../ixrt_common/config/MOBILENET_V2_CONFIG
```

### FP16

```bash
# Test ACC
bash script/infer_mobilenet_v2_fp16_accuracy.sh
# Test FPS
bash script/infer_mobilenet_v2_fp16_performance.sh
```

### INT8

```bash
# Test ACC
bash script/infer_mobilenet_v2_int8_accuracy.sh
# Test FPS
bash script/infer_mobilenet_v2_int8_performance.sh
```

## Model Results

| Model       | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
| ----------- | --------- | --------- | ------- | -------- | -------- |
| MobileNetV2 | 32        | FP16      | 4835.19 | 0.7186   | 0.90316  |

## Refereneces

- [Paper](https://arxiv.org/abs/1801.04381)
