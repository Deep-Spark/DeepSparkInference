# SqueezeNet 1.0

## Description
SqueezeNet 1.0 is a deep learning model for image classification, designed to be lightweight and efficient for deployment on resource-constrained devices. 

It was developed by researchers at DeepScale and released in 2016.

## Setup

### Install

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-dev

pip3 install tqdm
pip3 install onnxsim
pip3 install opencv-python==4.6.0.66
```
### Download 
Pretrained model: https://download.pytorch.org/models/squeezenet1_0-b66bff10.pth

Dataset: https://www.image-net.org/download.php to download the validation dataset.

### Model Conversion 
```
mkdir checkpoints 
python3 export_onnx.py --origin_model  /path/to/squeezenet1_0-b66bff10.pth --output_model checkpoints/squeezenetv10.onnx
```

## Inference
```
export PROJ_DIR=./
export DATASETS_DIR=/path/to/imagenet_val/
export CHECKPOINTS_DIR=./checkpoints
export RUN_DIR=./
export CONFIG_DIR=config/SQUEEZENET_V10_CONFIG

```
### FP16

```bash
# Accuracy
bash scripts/infer_squeezenet_v10_fp16_accuracy.sh
# Performance
bash scripts/infer_squeezenet_v10_fp16_performance.sh
```

### INT8

```bash
# Accuracy
bash scripts/infer_squeezenet_v10_int8_accuracy.sh
# Performance
bash scripts/infer_squeezenet_v10_int8_performance.sh
```

## Results 
Model          |BatchSize  |Precision |FPS      |Top-1(%)  |Top-5(%)
---------------|-----------|----------|---------|----------|--------
SqueezeNet 1.0 |    32     |   FP16   | 7740.26 |  58.07   | 80.43
SqueezeNet 1.0 |    32     |   INT8   | 8871.93 |  55.10   | 79.21

