# EfficientNet B5 (IGIE)

## Model Description

EfficientNet B5 is an efficient convolutional network model designed through a compound scaling strategy and NAS optimization, offering a good balance between high accuracy and computational efficiency across various vision tasks.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| MR-V100 | 4.2.0     |  25.06  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/efficientnet_b5_lukemelas-1a07897c.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
pip3 install -r requirements.txt
```

### Model Conversion

```bash
python3 export.py --weight efficientnet_b5_lukemelas-1a07897c.pth --output efficientnet_b5.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
```

### FP16

```bash
# Accuracy
bash scripts/infer_efficientnet_b5_fp16_accuracy.sh
# Performance
bash scripts/infer_efficientnet_b5_fp16_performance.sh
```

## Model Results

| Model           | BatchSize | Precision | FPS      | Top-1(%) | Top-5(%) |
| --------------- | --------- | --------- | -------- | -------- | -------- |
| Efficientnet_b5 | 32        | FP16      | 623.729  | 73.121   | 90.913   |
