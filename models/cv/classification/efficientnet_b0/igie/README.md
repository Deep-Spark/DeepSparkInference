# EfficientNet B0 (IGIE)

## Model Description

EfficientNet-B0 is a lightweight yet highly efficient convolutional neural network architecture. It is part of the EfficientNet family, known for its superior performance in balancing model size and accuracy. Developed with a focus on resource efficiency, EfficientNet-B0 achieves remarkable results across various computer vision tasks, including image classification and feature extraction.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.2.0     |  25.03  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
pip3 install -r ../../igie_common/requirements.txt
```

### Model Conversion

```bash
python3 ../../igie_common/export.py --model-name efficientnet_b0 --weight efficientnet_b0_rwightman-7f5810bc.pth --output efficientnet_b0.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
export RUN_DIR=../../igie_common/
```

### FP16

```bash
# Accuracy
bash scripts/infer_efficientnet_b0_fp16_accuracy.sh
# Performance
bash scripts/infer_efficientnet_b0_fp16_performance.sh
```

## Model Results

| Model           | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
|-----------------|-----------|-----------|---------|----------|----------|
| EfficientNet_B0 | 32        | FP16      | 2596.60 | 77.639   | 93.540   |
