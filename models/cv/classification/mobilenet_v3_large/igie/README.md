# MobileNetV3_Large (IGIE)

## Model Description

MobileNetV3_Large builds upon the success of its predecessors by incorporating several innovative design strategies to enhance performance. It features larger model capacity and computational resources compared to MobileNetV3_Small, allowing for deeper network architectures and more complex feature representations.

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
pip3 install -r requirements.txt
```

### Model Conversion

```bash
python3 export.py --weight mobilenet_v3_large-8738ca79.pth --output mobilenetv3_large.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
```

### FP16

```bash
# Accuracy
bash scripts/infer_mobilenet_v3_large_fp16_accuracy.sh
# Performance
bash scripts/infer_mobilenet_v3_large_fp16_performance.sh
```

## Model Results

| Model             | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
|-------------------|-----------|-----------|---------|----------|----------|
| MobileNetV3_Large | 32        | FP16      | 3644.08 | 74.042   | 91.303   |
