# VGG16

## Description

VGG16 is a convolutional neural network (CNN) architecture designed for image classification tasks.The architecture of VGG16 is characterized by its simplicity and uniform structure. It consists of 16 convolutional and fully connected layers, organized into five blocks, with the convolutional layers using small 3x3 filters.

## Setup

### Install
```
pip3 install onnx
pip3 install tqdm
```

### Download

Pretrained model: <https://download.pytorch.org/models/vgg16-397923af.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Model Conversion
```bash
python3 export.py --weight vgg16-397923af.pth --output vgg16.onnx
```

## Inference
```bash
export DATASETS_DIR=/Path/to/imagenet_val/
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

## Results

Model   |BatchSize  |Precision |FPS       |Top-1(%)  |Top-5(%)
--------|-----------|----------|----------|----------|--------
VGG16   |    32     |   FP16   | 1830.53  |  71.55   | 90.37
VGG16   |    32     |   INT8   | 3528.01  |  71.53   | 90.32
