# VGG19 (IGIE)

## Model Description

VGG19 is a member of the VGG network family, proposed by the Visual Geometry Group at the University of Oxford, originally designed for the ImageNet image classification task. Known for its deep structure and simple convolutional module design, VGG19 is one of the deepest models in the series, featuring 19 weight layers (16 convolutional layers and 3 fully connected layers). Its depth and regular network design achieved outstanding classification performance at the time.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| MR-V100 | 4.2.0     |  25.06  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/vgg19-dcbb9e9d.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
pip3 install -r requirements.txt
```

### Model Conversion

```bash
python3 export.py --weight vgg19-dcbb9e9d.pth --output vgg19.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
```

### FP16

```bash
# Accuracy
bash scripts/infer_vgg19_fp16_accuracy.sh
# Performance
bash scripts/infer_vgg19_fp16_performance.sh
```

## Model Results

| Model | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
|-------|-----------|-----------|---------|----------|----------|
| VGG19 | 32        | FP16      | 1654.54 | 72.353   | 90.853   |
