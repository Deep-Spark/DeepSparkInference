# ResNet50 (IGIE)

## Model Description

ResNet-50 is a convolutional neural network architecture that belongs to the ResNet.The key innovation in ResNet-50 is the introduction of residual blocks, which include shortcut connections (skip connections) to enable the flow of information directly from one layer to another. These shortcut connections help mitigate the vanishing gradient problem and facilitate the training of very deep networks.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.2.0     |  25.03  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/resnet50-0676ba61.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
pip3 install -r requirements.txt
```

### Model Conversion

```bash
python3 export.py --weight resnet50-0676ba61.pth --output resnet50.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
```

### FP16

```bash
# Accuracy
bash scripts/infer_resnet50_fp16_accuracy.sh
# Performance
bash scripts/infer_resnet50_fp16_performance.sh
```

### INT8

```bash
# Accuracy
bash scripts/infer_resnet50_int8_accuracy.sh
# Performance
bash scripts/infer_resnet50_int8_performance.sh
```

## Model Results

| Model    | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
|----------|-----------|-----------|---------|----------|----------|
| ResNet50 | 32        | FP16      | 4417.29 | 76.11    | 92.85    |
| ResNet50 | 32        | INT8      | 8628.61 | 75.72    | 92.71    |
