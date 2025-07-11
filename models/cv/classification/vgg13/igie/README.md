# VGG13 (IGIE)

## Model Description

VGG13 is a classic deep convolutional neural network model consisting of 13 convolutional layers and multiple pooling layers. It utilizes 3×3 small convolution kernels to extract image features and completes classification through fully connected layers. Known for its simple structure and high performance, it is well-suited for image classification tasks but requires significant computational resources due to its large parameter size.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.2.0     |  25.03  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/vgg13-19584684.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
pip3 install -r ../../igie_common/requirements.txt
```

### Model Conversion

```bash
python3 ../../igie_common/export.py --model-name vgg13 --weight vgg13-19584684.pth --output vgg13.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
export RUN_DIR=../../igie_common/
```

### FP16

```bash
# Accuracy
bash scripts/infer_vgg13_fp16_accuracy.sh
# Performance
bash scripts/infer_vgg13_fp16_performance.sh
```

## Model Results

| Model  | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
| :----: | :-------: | :-------: | :-----: | :------: | :------: |
| VGG13  | 32        | FP16      | 2598.51 | 69.894   | 89.233   |
