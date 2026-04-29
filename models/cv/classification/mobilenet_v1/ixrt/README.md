# MobileNetV1 (ixRT)

## Model Description

MobileNetV1 is a efficient model architecture using depthwise separable convolutions. It is designed to efficiently maximize accuracy while being mindful of the tight resource constraints.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.4.0 | 26.06 |

## Model Preparation

### Prepare Resources

Pretrained model: <http://files.deepspark.org.cn:880/deepspark/data/checkpoints/mobilenet_v1.onnx>

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
cd checkpoints
wget http://files.deepspark.org.cn:880/deepspark/data/checkpoints/mobilenet_v1.onnx
```

## Model Inference

```bash
export PROJ_DIR=./
export DATASETS_DIR=/path/to/imagenet_val/
export CHECKPOINTS_DIR=./checkpoints
export RUN_DIR=../../ixrt_common/
export CONFIG_DIR=../../ixrt_common/config/MOBILENET_V1_CONFIG
```

### FP16

```bash
# Test ACC
bash scripts/infer_mobilenet_v1_fp16_accuracy.sh
# Test FPS
bash scripts/infer_mobilenet_v1_fp16_performance.sh
```

### INT8

```bash
# Test ACC
bash scripts/infer_mobilenet_v1_int8_accuracy.sh
# Test FPS
bash scripts/infer_mobilenet_v1_int8_performance.sh
```

## Model Results

| Model       | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
| ----------- | --------- | --------- | ------- | -------- | -------- |
| MobileNetV1 | 32        | FP16      | 13862.317   | 71.6  | 90.3  |
| MobileNetV1 | 32        | INT8      | 17485.601  | 70.9   | 89.9 |