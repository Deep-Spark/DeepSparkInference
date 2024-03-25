# AlexNet

## Description

AlexNet, developed by Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton, is a groundbreaking convolutional neural network (CNN) architecture that achieved remarkable success in the 2012 ImageNet Large Scale Visual Recognition Challenge (ILSVRC). This neural network comprises eight layers, incorporating five convolutional layers and three fully connected layers. The architecture employs the Rectified Linear Unit (ReLU) activation function to introduce non-linearity, allowing the model to learn complex features from input images.

## Setup

### Install
```
pip3 install onnx
pip3 install tqdm
```

### Download

Pretrained model: <https://download.pytorch.org/models/alexnet-owt-7be5be79.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Model Conversion
```bash
python3 export.py --weight alexnet-owt-7be5be79.pth --output alexnet.onnx
```

## Inference
```bash
export DATASETS_DIR=/Path/to/imagenet_val/
```
### FP16

```bash
# Accuracy
bash scripts/infer_alexnet_fp16_accuracy.sh
# Performance
bash scripts/infer_alexnet_fp16_performance.sh
```

### INT8
```bash
# Accuracy
bash scripts/infer_alexnet_int8_accuracy.sh
# Performance
bash scripts/infer_alexnet_int8_performance.sh
```

## Results

Model   |BatchSize  |Precision |FPS       |Top-1(%) |Top-5(%)
--------|-----------|----------|----------|---------|--------
AlexNet |    32     |   FP16   | 20456.16 |  56.53  | 79.05
AlexNet |    32     |   INT8   | 22465.46 |  55.96  | 78.83
