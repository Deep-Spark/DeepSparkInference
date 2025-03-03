# ResNeSt50

## Description

ResNeSt50 is a deep convolutional neural network model based on the ResNeSt architecture, specifically designed to enhance performance in visual recognition tasks such as image classification, object detection, instance segmentation, and semantic segmentation. ResNeSt stands for Split-Attention Networks, a modular network architecture that leverages channel-wise attention mechanisms across different network branches to capture cross-feature interactions and learn diverse representations.

## Setup

### Install

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

pip3 install -r requirements.txt
```

### Download

Pretrained model: <https://github.com/zhanghang1989/ResNeSt/releases/download/weights_step1/resnest50-528c19ca.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Model Conversion

```bash
# export onnx model
python3 export.py --weight resnest50-528c19ca.pth --output resnest50.onnx

# Use onnxsim optimize onnx model
onnxsim resnest50.onnx resnest50_opt.onnx

```

## Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
```

### FP16

```bash
# Accuracy
bash scripts/infer_resnest50_fp16_accuracy.sh
# Performance
bash scripts/infer_resnest50_fp16_performance.sh
```

## Results

Model     |BatchSize  |Precision |FPS       |Top-1(%)  |Top-5(%)
----------|-----------|----------|----------|----------|--------
ResNeSt50 |    32     |   FP16   | 344.453  |  80.93   | 95.347

## Reference

ResNeSt50: <https://github.com/zhanghang1989/ResNeSt>
