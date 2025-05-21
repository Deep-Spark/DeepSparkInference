# EfficientNet B1 (IGIE)

## Model Description

EfficientNet B1 is a convolutional neural network architecture that falls under the EfficientNet family, known for its remarkable balance between model size and performance. Introduced as part of the EfficientNet series, EfficientNet B1 offers a compact yet powerful solution for various computer vision tasks, including image classification, object detection and segmentation.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.2.0     |  25.03  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/efficientnet_b1-c27df63c.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
pip3 install -r ../../igie_common/requirements.txt
```

### Model Conversion

```bash
python3 ../../igie_common/export.py --model-name efficientnet_b1 --weight efficientnet_b1-c27df63c.pth --output efficientnet_b1.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
export RUN_DIR=../../igie_common/
```

### FP16

```bash
# Accuracy
bash scripts/infer_efficientnet_b1_fp16_accuracy.sh
# Performance
bash scripts/infer_efficientnet_b1_fp16_performance.sh
```

## Model Results

| Model           | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
|-----------------|-----------|-----------|---------|----------|----------|
| EfficientNet B1 | 32        | FP16      | 1292.31 | 78.823   | 94.494   |
