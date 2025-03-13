# MNASNet0_5 (IGIE)

## Model Description

MNASNet0_5 is a neural network architecture optimized for mobile devices, designed through neural architecture search technology. It is characterized by high efficiency and excellent accuracy, offering 50% higher accuracy than MobileNetV2 while maintaining low latency and memory usage. MNASNet0_5 widely uses depthwise separable convolutions, supports multi-scale inputs, and demonstrates good robustness, making it suitable for real-time image recognition tasks in resource-constrained environments.

## Supported Environments

| Iluvatar GPU | IXUCA SDK |
|--------------|-----------|
| MR-V100      | 4.2.0     |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/mnasnet0.5_top1_67.823-3ffadce67e.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
pip3 install -r requirements.txt
```

### Model Conversion

```bash
python3 export.py --weight mnasnet0.5_top1_67.823-3ffadce67e.pth --output mnasnet0_5.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
```

### FP16

```bash
# Accuracy
bash scripts/infer_mnasnet0_5_fp16_accuracy.sh
# Performance
bash scripts/infer_mnasnet0_5_fp16_performance.sh
```

## Model Results

| Model             | BatchSize | Precision | FPS      | Top-1(%) | Top-5(%) |
| ----------------- | --------- | --------- | -------- | -------- | -------- |
| MnasNet0_5        | 32        | FP16      | 7933.980 | 67.748   |  87.452  |
