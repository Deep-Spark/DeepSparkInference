# ResNext101_64x4d

## Description

The ResNeXt101_64x4d is a deep learning model based on the deep residual network architecture, which enhances performance and efficiency through the use of grouped convolutions. With a depth of 101 layers and 64 filter groups, it is particularly suited for complex image recognition tasks. While maintaining excellent accuracy, it can adapt to various input sizes

## Setup

### Install

```bash
pip3 install onnx
pip3 install tqdm
```

### Download

Pretrained model: <https://download.pytorch.org/models/resnext101_64x4d-173b62eb.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Model Conversion

```bash
python3 export.py --weight resnext101_64x4d-173b62eb.pth --output resnext101_64x4d.onnx
```

## Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
```

### FP16

```bash
# Accuracy
bash scripts/infer_resnext101_64x4d_fp16_accuracy.sh
# Performance
bash scripts/infer_resnext101_64x4d_fp16_performance.sh
```

## Results

Model           |BatchSize  |Precision |FPS      |Top-1(%)  |Top-5(%)
----------------|-----------|----------|---------|----------|--------
ResNext101_64x4d|    32     |   FP16   | 663.13  |  82.953  | 96.221