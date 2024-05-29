# VGG16

## Description
VGG16 is a deep convolutional neural network model developed by the Visual Geometry Group at the University of Oxford. 
It finished second in the 2014 ImageNet Massive Visual Identity Challenge (ILSVRC).

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

Dataset: https://www.image-net.org/download.php to download the validation dataset.

### Model Conversion
```
mkdir checkpoints 
python3 export_onnx.py --output_model checkpoints/vgg16.onnx
```

## Inference
```
export PROJ_DIR=./
export DATASETS_DIR=/path/to/imagenet_val/
export CHECKPOINTS_DIR=./checkpoints
export RUN_DIR=./
export CONFIG_DIR=config/VGG16_CONFIG
```
### FP16

```bash
# Accuracy
bash scripts/infer_vgg16_fp16_accuracy.sh
# Performance
bash scripts/infer_vgg16_fp16_performance.sh
```

### INT8
```bash
# Accuracy
bash scripts/infer_vgg16_int8_accuracy.sh
# Performance
bash scripts/infer_vgg16_int8_performance.sh
```

## Results 

Model |BatchSize  |Precision |FPS      |Top-1(%) |Top-5(%)
------|-----------|----------|---------|---------|--------
VGG16 |    32     |   FP16   | 1777.85 |  71.57  | 90.40
VGG16 |    32     |   INT8   | 4451.80 |  71.47  | 90.35

