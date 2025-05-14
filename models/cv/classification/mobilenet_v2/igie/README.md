# MobileNetV2 (IGIE)

## Model Description

MobileNetV2 is an improvement on V1. Its new ideas include Linear Bottleneck and Inverted Residuals, and is based on an inverted residual structure where the input and output of the residual block are thin bottleneck layers. The intermediate expansion layer uses lightweight depthwise convolutions to filter features as a source of non-linearity.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.2.0     |  25.03  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/mobilenet_v2-7ebf99e0.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
pip3 install -r requirements.txt
```

### Model Conversion

```bash
python3 export.py --weight mobilenet_v2-7ebf99e0.pth --output mobilenet_v2.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
```

### FP16

```bash
# Accuracy
bash scripts/infer_mobilenet_v2_fp16_accuracy.sh
# Performance
bash scripts/infer_mobilenet_v2_fp16_performance.sh
```

### INT8

```bash
# Accuracy
bash scripts/infer_mobilenet_v2_int8_accuracy.sh
# Performance
bash scripts/infer_mobilenet_v2_int8_performance.sh
```

## Model Results

| Model       | BatchSize | Precision | FPS      | Top-1(%) | Top-5(%) |
|-------------|-----------|-----------|----------|----------|----------|
| MobileNetV2 | 32        | FP16      | 6910.65  | 71.96    | 90.60    |
| MobileNetV2 | 32        | INT8      | 8155.362 | 71.48    | 90.47    |
