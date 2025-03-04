# EfficientNet_v2_s

## Model Description

EfficientNetV2 S is an optimized model in the EfficientNetV2 series, which was developed by Google researchers. It continues the legacy of the EfficientNet family, focusing on advancing the state-of-the-art in accuracy and efficiency through advanced scaling techniques and architectural innovations.

## Model Preparation

### Install Dependencies

```bash
pip3 install -r requirements.txt
```

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/efficientnet_v2_s-dd5fe13b.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

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
| Efficientnet_v2_s | 32        | FP16      | 2020.388 | 81.312   | 95.288   |
