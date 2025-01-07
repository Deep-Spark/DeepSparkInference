# CSPResNet50

## Description

Neural networks have enabled state-of-the-art approaches to achieve incredible results on computer vision tasks such as object detection.
CSPResNet50 is the one of best models.

## Setup

### Install

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-dev

pip3 install -r requirements.txt
```

### Download

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Model Conversion

```bash
mkdir checkpoints 
git clone -b v0.24.0 https://github.com/open-mmlab/mmpretrain.git

python3 export_onnx.py   \
    --config_file ./mmpretrain/configs/cspnet/cspresnet50_8xb32_in1k.py  \
    --checkpoint_file  https://download.openmmlab.com/mmclassification/v0/cspnet/cspresnet50_3rdparty_8xb32_in1k_20220329-dd6dddfb.pth \
    --output_model ./checkpoints/cspresnet50.onnx
```

## Inference

```bash
export PROJ_DIR=./
export DATASETS_DIR=/path/to/imagenet_val
export CHECKPOINTS_DIR=./checkpoints
export RUN_DIR=./
export CONFIG_DIR=config/CSPRESNET50_CONFIG

```

### FP16

```bash
# Accuracy
bash scripts/infer_cspresnet50_fp16_accuracy.sh
# Performance
bash scripts/infer_cspresnet50_fp16_performance.sh
```

### INT8

```bash
# Accuracy
bash scripts/infer_cspresnet50_int8_accuracy.sh
# Performance
bash scripts/infer_cspresnet50_int8_performance.sh
```

## Results

Model       |BatchSize  |Precision |FPS      |Top-1(%)  |Top-5(%)
------------|-----------|----------|---------|----------|--------
CSPResNet50 |    32     |   FP16   | 4555.95 |  78.51   | 94.17
CSPResNet50 |    32     |   INT8   | 8801.94 |  78.15   | 93.95
