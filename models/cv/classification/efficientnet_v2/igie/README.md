# EfficientNetV2-M

## Description

EfficientNetV2 M is an optimized model in the EfficientNetV2 series, which was developed by Google researchers. It continues the legacy of the EfficientNet family, focusing on advancing the state-of-the-art in accuracy and efficiency through advanced scaling techniques and architectural innovations.

## Setup

### Install

```bash
pip3 install -r requirements.txt
```

### Download

Pretrained model: <https://download.pytorch.org/models/efficientnet_v2_m-dc08266a.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Model Conversion

```bash
python3 export.py --weight efficientnet_v2_m-dc08266a.pth --output efficientnet_v2_m.onnx
```

## Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
```

### FP16

```bash
# Accuracy
bash scripts/infer_efficientnet_v2_m_fp16_accuracy.sh
# Performance
bash scripts/infer_efficientnet_v2_m_fp16_performance.sh
```

## Results

| Model            | BatchSize | Precision | FPS      | Top-1(%) | Top-5(%) |
| ---------------- | --------- | --------- | -------- | -------- | -------- |
| EfficientNetV2-M | 32        | FP16      | 1104.846 | 79.635   | 94.456   |
