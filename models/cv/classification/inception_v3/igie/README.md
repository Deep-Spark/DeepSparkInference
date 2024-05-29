# Inception V3

## Description

Inception v3 is a convolutional neural network architecture designed for image recognition and classification tasks. Developed by Google, it represents an evolution of the earlier Inception models. Inception v3 is characterized by its deep architecture, featuring multiple layers with various filter sizes and efficient use of computational resources. The network employs techniques like factorized convolutions and batch normalization to enhance training stability and accelerate convergence.

## Setup

### Install

```bash
pip3 install onnx
pip3 install tqdm
```

### Download

Pretrained model: <https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Model Conversion

```bash
python3 export.py --weight inception_v3_google-0cc3c7bd.pth --output inception_v3.onnx
```

## Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
```

### FP16

```bash
# Accuracy
bash scripts/infer_inception_v3_fp16_accuracy.sh
# Performance
bash scripts/infer_inception_v3_fp16_performance.sh
```

### INT8

```bash
# Accuracy
bash scripts/infer_inception_v3_int8_accuracy.sh
# Performance
bash scripts/infer_inception_v3_int8_performance.sh
```

## Results

Model        |BatchSize  |Precision |FPS       |Top-1(%)  |Top-5(%)
-------------|-----------|----------|----------|----------|--------
Inception_v3 |    32     |   FP16   | 3557.25  |  69.848  | 88.858
Inception_v3 |    32     |   INT8   | 3631.80  |  69.022  | 88.412
