# EfficientNet B4 (IGIE)

## Model Description

EfficientNet B4 is a high-performance convolutional neural network model introduced in Google's paper "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." It is part of the EfficientNet family, which leverages compound scaling to balance depth, width, and input resolution for better accuracy and efficiency.

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/efficientnet_b4_rwightman-23ab8bcd.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
pip3 install -r requirements.txt
```

### Model Conversion

```bash
python3 export.py --weight efficientnet_b4_rwightman-23ab8bcd.pth --output efficientnet_b4.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
```

### FP16

```bash
# Accuracy
bash scripts/infer_efficientnet_b4_fp16_accuracy.sh
# Performance
bash scripts/infer_efficientnet_b4_fp16_performance.sh
```

## Model Results

| Model           | BatchSize | Precision | FPS      | Top-1(%) | Top-5(%) |
| --------------- | --------- | --------- | -------- | -------- | -------- |
| Efficientnet_b4 | 32        | FP16      | 991.397  | 79.261   | 94.496   |
