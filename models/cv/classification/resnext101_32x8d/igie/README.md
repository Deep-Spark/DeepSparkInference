# ResNext101_32x8d

## Description

ResNeXt101_32x8d is a deep convolutional neural network introduced in the paper "Aggregated Residual Transformations for Deep Neural Networks." It enhances the traditional ResNet architecture by incorporating group convolutions, offering a new dimension for scaling network capacity through "cardinality" (the number of groups) rather than merely increasing depth or width.The model consists of 101 layers and uses a configuration of 32 groups, each with a width of 8 channels. This design improves feature extraction while maintaining computational efficiency.

## Setup

### Install

```bash
pip3 install -r requirements.txt
```

### Download

Pretrained model: <https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Model Conversion

```bash
python3 export.py --weight resnext101_32x8d-8ba56ff5.pth --output resnext101_32x8d.onnx
```

## Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
```

### FP16

```bash
# Accuracy
bash scripts/infer_resnext101_32x8d_fp16_accuracy.sh
# Performance
bash scripts/infer_resnext101_32x8d_fp16_performance.sh
```

## Results

| Model            | BatchSize | Precision | FPS    | Top-1(%) | Top-5(%) |
| ---------------- | --------- | --------- | ------ | -------- | -------- |
| ResNext101_32x8d | 32        | FP16      | 825.78 | 79.277   | 94.498   |
