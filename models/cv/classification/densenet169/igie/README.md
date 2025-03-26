# DenseNet169 (IGIE)

## Model Description

DenseNet-169 is a variant of the Dense Convolutional Network (DenseNet) architecture, characterized by its 169 layers and a growth rate of 32. This network leverages the dense connectivity pattern, where each layer is connected to every other layer in a feed-forward fashion, resulting in a substantial increase in the number of direct connections compared to traditional convolutional networks. This connectivity pattern facilitates the reuse of features and enhances the flow of information and gradients throughout the network, which is particularly beneficial for deep architectures.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| MR-V100 | 4.2.0     |  25.03  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/densenet169-b2777c0a.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
pip3 install -r requirements.txt
```

### Model Conversion

```bash
python3 export.py --weight densenet169-b2777c0a.pth --output densenet169.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
```

### FP16

```bash
# Accuracy
bash scripts/infer_densenet169_fp16_accuracy.sh
# Performance
bash scripts/infer_densenet169_fp16_performance.sh
```

## Model Results

| Model       | BatchSize | Precision | FPS      | Top-1(%) | Top-5(%) |
| ----------- | --------- | --------- | -------- | -------- | -------- |
| DenseNet169 | 32        | FP16      | 1384.649 | 75.548   | 92.778   |
