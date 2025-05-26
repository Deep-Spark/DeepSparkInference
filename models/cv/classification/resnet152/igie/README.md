# ResNet152 (IGIE)

## Model Description

ResNet152 is a convolutional neural network architecture that is part of the ResNet (Residual Network) family, Comprising 152 layers, At the core of ResNet152 is the innovative residual learning framework, which addresses the challenges associated with training very deep neural networks.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.2.0     |  25.03  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/resnet152-394f9c45.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
pip3 install -r ../../igie_common/requirements.txt
```

### Model Conversion

```bash
python3 ../../igie_common/export.py --model-name resnet152 --weight resnet152-394f9c45.pth --output resnet152.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
export RUN_DIR=../../igie_common/
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
| :----: | :----: | :----: | :----: | :----: | :----: |
| ResNet152 | 32        | FP16      | 1768.348 | 78.285   | 94.022   |
| ResNet152 | 32        | INT8      | 3864.913 | 77.637   | 93.728   |
