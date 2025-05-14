# EfficientNet_v2_s (IGIE)

## Model Description

EfficientNetV2 S is an optimized model in the EfficientNetV2 series, which was developed by Google researchers. It continues the legacy of the EfficientNet family, focusing on advancing the state-of-the-art in accuracy and efficiency through advanced scaling techniques and architectural innovations.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.2.0     |  25.03  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/efficientnet_v2_s-dd5fe13b.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
pip3 install -r requirements.txt
```

### Model Conversion

```bash
python3 export.py --weight efficientnet_v2_s-dd5fe13b.pth --output efficientnet_v2_s.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
```

### FP16

```bash
# Accuracy
bash scripts/infer_efficientnet_v2_s_fp16_accuracy.sh
# Performance
bash scripts/infer_efficientnet_v2_s_fp16_performance.sh
```

## Model Results

| Model             | BatchSize | Precision | FPS      | Top-1(%) | Top-5(%) |
| ----------------- | --------- | --------- | -------- | -------- | -------- |
| Efficientnet_v2_s | 32        | FP16      | 2357.457 | 81.290   | 95.242   |
