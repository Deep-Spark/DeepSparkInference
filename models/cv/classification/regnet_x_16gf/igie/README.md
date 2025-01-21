# RegNet_x_16gf

## Description

RegNet_x_16gf is a deep convolutional neural network from the RegNet family, introduced in the paper "Designing Network Design Spaces" by Facebook AI. RegNet models emphasize simplicity, efficiency, and scalability, and they systematically explore design spaces to achieve optimal performance.The x in RegNet_x_16gf indicates it belongs to the RegNetX series, which focuses on optimizing network width and depth, while 16gf refers to its computational complexity of approximately 16 GFLOPs. The model features linear width scaling, group convolutions, and bottleneck blocks, providing high accuracy while maintaining computational efficiency.


## Setup

### Install

```bash
pip3 install -r requirements.txt
```

### Download

Pretrained model: <https://download.pytorch.org/models/regnet_x_16gf-2007eb11.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Model Conversion

```bash
python3 export.py --weight regnet_x_16gf-2007eb11.pth --output regnet_x_16gf.onnx
```

## Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
```

### FP16

```bash
# Accuracy
bash scripts/infer_regnet_x_16gf_fp16_accuracy.sh
# Performance
bash scripts/infer_regnet_x_16gf_fp16_performance.sh
```

## Results

Model             |BatchSize  |Precision |FPS      |Top-1(%) |Top-5(%)
------------------|-----------|----------|---------|---------|--------
RegNet_x_16gf     |    32     |   FP16   | 970.928 | 80.028  | 94.922