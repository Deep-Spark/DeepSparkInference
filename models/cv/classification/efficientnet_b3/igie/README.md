# EfficientNet B3 (IGIE)

## Model Description

EfficientNet B3 is a member of the EfficientNet family, a series of convolutional neural network architectures that are designed to achieve excellent accuracy and efficiency. Introduced by researchers at Google, EfficientNets utilize the compound scaling method, which uniformly scales the depth, width, and resolution of the network to improve accuracy and efficiency.

## Supported Environments

| Iluvatar GPU | IXUCA SDK |
|--------------|-----------|
| MR-V100      | 4.2.0     |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/efficientnet_b3_rwightman-b3899882.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
pip3 install -r requirements.txt
```

### Model Conversion

```bash
python3 export.py --weight efficientnet_b3_rwightman-b3899882.pth --output efficientnet_b3.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
```

### FP16

```bash
# Accuracy
bash scripts/infer_efficientnet_b3_fp16_accuracy.sh
# Performance
bash scripts/infer_efficientnet_b3_fp16_performance.sh
```

## Model Results

| Model           | BatchSize | Precision | FPS      | Top-1(%) | Top-5(%) |
| --------------- | --------- | --------- | -------- | -------- | -------- |
| Efficientnet_b3 | 32        | FP16      | 1144.391 | 78.503   | 94.340   |
