# EfficientNet B0

## Description
EfficientNet B0 is a convolutional neural network architecture that belongs to the EfficientNet family, which was introduced by Mingxing Tan and Quoc V. Le in their paper "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." The EfficientNet family is known for achieving state-of-the-art performance on various computer vision tasks while being more computationally efficient than many existing models.

## Setup

### Install
```bash
yum install mesa-libGL
pip3 install tqdm
pip3 install onnx
pip3 install onnxsim
pip3 install tabulate
pip3 install ppq==0.6.6
pip install protobuf==3.20.3 pycuda
```

### Download
Dataset: <https://www.image-net.org/download.php> to download the validation dataset.


## Inference
```bash
export DATASETS_DIR=/path/to/imagenet_val/
```
### FP16

```bash
# Accuracy
bash scripts/infer_efficientnet_b0_fp16_accuracy.sh
# Performance
bash scripts/infer_efficientnet_b0_fp16_performance.sh
```

### INT8
```bash
# Accuracy
bash scripts/infer_efficientnet_b0_int8_accuracy.sh
# Performance
bash scripts/infer_efficientnet_b0_int8_performance.sh
```

## Results 
Model           |BatchSize  |Precision |FPS       |Top-1(%)  |Top-5(%)
----------------|-----------|----------|----------|----------|--------
EfficientNet_B0 |    32     |   FP16   | 2325.54  |  77.66   | 93.58
EfficientNet_B0 |    32     |   INT8   | 2666.00  |  74.27   | 91.85
