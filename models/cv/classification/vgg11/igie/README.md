# VGG11

## Model Description

VGG11 is a deep convolutional neural network introduced by the Visual Geometry Group at the University of Oxford in the paper "Very Deep Convolutional Networks for Large-Scale Image Recognition." The model consists of 11 layers with trainable weights, including 8 convolutional layers and 3 fully connected layers. It employs small 3x3 convolutional kernels and 2x2 max-pooling layers to extract hierarchical features from input images. The ReLU activation function is used throughout the network to enhance non-linearity and mitigate the vanishing gradient problem.

## Model Preparation

### Install Dependencies

```bash
pip3 install -r requirements.txt
```

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/vgg11-8a719046.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Model Conversion

```bash
python3 export.py --weight vgg11-8a719046.pth --output vgg11.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
```

### FP16

```bash
# Accuracy
bash scripts/infer_vgg11_fp16_accuracy.sh
# Performance
bash scripts/infer_vgg11_fp16_performance.sh
```

## Model Results

Model   |BatchSize  |Precision |FPS       |Top-1(%)  |Top-5(%)
--------|-----------|----------|----------|----------|--------
VGG11   |    32     |   FP16   | 3872.86  |  69.03   | 88.6
