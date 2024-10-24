# ResNet101

## Description

ResNet101 is a convolutional neural network architecture that belongs to the ResNet (Residual Network) family.With a total of 101 layers, ResNet101 comprises multiple residual blocks, each containing convolutional layers with batch normalization and rectified linear unit (ReLU) activations. These residual blocks allow the network to effectively capture complex features at different levels of abstraction, leading to superior performance on image recognition tasks.

## Setup

### Install

```bash
pip3 install -r requirements.txt
```

### Download

Pretrained model: <https://download.pytorch.org/models/resnet101-63fe2227.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Model Conversion

```bash
python3 export.py --weight resnet101-63fe2227.pth --output resnet101.onnx
```

## Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
```

### FP16

```bash
# Accuracy
bash scripts/infer_resnet101_fp16_accuracy.sh
# Performance
bash scripts/infer_resnet101_fp16_performance.sh
```

### INT8

```bash
# Accuracy
bash scripts/infer_resnet101_int8_accuracy.sh
# Performance
bash scripts/infer_resnet101_int8_performance.sh
```

## Results

Model     |BatchSize  |Precision |FPS       |Top-1(%)  |Top-5(%)
----------|-----------|----------|----------|----------|--------
ResNet101 |    32     |   FP16   | 2507.074 |  77.331  |  93.520
ResNet101 |    32     |   INT8   | 5458.890 |  76.719  |  93.348
