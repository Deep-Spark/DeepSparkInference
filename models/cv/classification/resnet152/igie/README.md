# ResNet152 (IGIE)

## Model Description

ResNet152 is a convolutional neural network architecture that is part of the ResNet (Residual Network) family, Comprising 152 layers, At the core of ResNet152 is the innovative residual learning framework, which addresses the challenges associated with training very deep neural networks.

## Supported Environments

| Iluvatar GPU | IXUCA SDK |
|--------------|-----------|
| MR-V100      | 4.2.0     |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/resnet152-394f9c45.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
pip3 install -r requirements.txt
```

### Model Conversion

```bash
python3 export.py --weight resnet152-394f9c45.pth --output resnet152.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
```

### FP16

```bash
# Accuracy
bash scripts/infer_resnet152_fp16_accuracy.sh
# Performance
bash scripts/infer_resnet152_fp16_performance.sh
```

### INT8

```bash
# Accuracy
bash scripts/infer_resnet152_int8_accuracy.sh
# Performance
bash scripts/infer_resnet152_int8_performance.sh
```

## Model Results

| Model     | BatchSize | Precision | FPS      | Top-1(%) | Top-5(%) |
|-----------|-----------|-----------|----------|----------|----------|
| ResNet152 | 32        | FP16      | 1768.348 | 78.285   | 94.022   |
| ResNet152 | 32        | INT8      | 3864.913 | 77.637   | 93.728   |
