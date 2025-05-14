# ResNext50_32x4d (IGIE)

## Model Description

The ResNeXt50_32x4d model is a convolutional neural network architecture designed for image classification tasks. It is an extension of the ResNet (Residual Network) architecture, incorporating the concept of cardinality to enhance model performance.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.2.0     |  25.03  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
pip3 install -r requirements.txt
```

### Model Conversion

```bash
python3 export.py --weight resnext50_32x4d-7cdf4587.pth --output resnext50_32x4d.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
```

### FP16

```bash
# Accuracy
bash scripts/infer_resnext50_32x4d_fp16_accuracy.sh
# Performance
bash scripts/infer_resnext50_32x4d_fp16_performance.sh
```

## Model Results

| Model           | BatchSize | Precision | FPS    | Top-1(%) | Top-5(%) |
|-----------------|-----------|-----------|--------|----------|----------|
| ResNext50_32x4d | 32        | FP16      | 273.20 | 77.601   | 93.656   |
