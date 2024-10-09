# ResNext50_32x4d

## Description

The ResNeXt50_32x4d model is a convolutional neural network architecture designed for image classification tasks. It is an extension of the ResNet (Residual Network) architecture, incorporating the concept of cardinality to enhance model performance.

## Setup

### Install

```bash
pip3 install onnx
pip3 install tqdm
pip3 install tqdm
pip3 install onnx
pip3 install onnxsim
pip3 install tabulate
pip3 install ppq
pip3 install cuda-python
```

### Download

Pretrained model: <https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Model Conversion

```bash
python3 export.py --weight resnext50_32x4d-7cdf4587.pth --output resnext50_32x4d.onnx
```

## Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
```

### FP16

```bash
# Accuracy
bash scripts/infer_resnext50_32x4d_fp16_accuracy.sh
# Performance
bash scripts/infer_resnext50_32x4d_fp16_performance.sh
```

## Results

Model           |BatchSize  |Precision |FPS      |Top-1(%)  |Top-5(%)
----------------|-----------|----------|---------|----------|--------
resnext50_32x4d |    32     |   FP16   | 417.01  |  77.614  | 93.686
