# VGG11 (IGIE)

## Model Description

VGG11 is a deep convolutional neural network introduced by the Visual Geometry Group at the University of Oxford in the paper "Very Deep Convolutional Networks for Large-Scale Image Recognition." The model consists of 11 layers with trainable weights, including 8 convolutional layers and 3 fully connected layers. It employs small 3x3 convolutional kernels and 2x2 max-pooling layers to extract hierarchical features from input images. The ReLU activation function is used throughout the network to enhance non-linearity and mitigate the vanishing gradient problem.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.2.0     |  25.03  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/vgg11-8a719046.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
pip3 install -r ../../igie_common/requirements.txt
```

### Model Conversion

```bash
python3 ../../igie_common/export.py --model-name vgg11 --weight vgg11-8a719046.pth --output vgg11.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
export RUN_DIR=../../igie_common/
```

### FP16

```bash
# Accuracy
bash scripts/infer_vgg11_fp16_accuracy.sh
# Performance
bash scripts/infer_vgg11_fp16_performance.sh
```

## Model Results

| Model | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
| :----: | :----: | :----: | :----: | :----: | :----: |
| VGG11 | 32        | FP16      | 3872.86 | 69.03    | 88.6     |
