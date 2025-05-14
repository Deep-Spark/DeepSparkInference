# MobileNetV3_Small (IGIE)

## Model Description

MobileNetV3_Small is a lightweight convolutional neural network architecture designed for efficient mobile and embedded devices. It is part of the MobileNet family, renowned for its compact size and high performance, making it ideal for applications with limited computational resources.The key focus of MobileNetV3_Small is to achieve a balance between model size, speed, and accuracy.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.2.0     |  25.03  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
pip3 install -r requirements.txt
```

### Model Conversion

```bash
python3 export.py --weight mobilenet_v3_small-047dcff4.pth --output mobilenetv3_small.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
```

### FP16

```bash
# Accuracy
bash scripts/infer_mobilenet_v3_fp16_accuracy.sh
# Performance
bash scripts/infer_mobilenet_v3_fp16_performance.sh
```

## Model Results

| Model             | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
|-------------------|-----------|-----------|---------|----------|----------|
| MobileNetV3_Small | 32        | FP16      | 6837.86 | 67.612   | 87.404   |
