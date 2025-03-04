# VGG16

## Model Description

VGG16 is a deep convolutional neural network model developed by the Visual Geometry Group at the University of Oxford.
It finished second in the 2014 ImageNet Massive Visual Identity Challenge (ILSVRC).

## Model Preparation

### Install Dependencies

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

pip3 install -r requirements.txt
```

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/vgg16-397923af.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Model Conversion

```bash
mkdir checkpoints 
python3 export_onnx.py --origin_model /path/to/vgg16-397923af.pth --output_model checkpoints/vgg16.onnx
```

## Model Inference

```bash
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

## Model Results

Model |BatchSize  |Precision |FPS      |Top-1(%) |Top-5(%)
------|-----------|----------|---------|---------|--------
VGG16 |    32     |   FP16   | 1777.85 |  71.57  | 90.40
VGG16 |    32     |   INT8   | 4451.80 |  71.47  | 90.35
