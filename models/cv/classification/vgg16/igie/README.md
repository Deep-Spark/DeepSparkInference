# VGG16 (IGIE)

## Model Description

VGG16 is a convolutional neural network (CNN) architecture designed for image classification tasks.The architecture of VGG16 is characterized by its simplicity and uniform structure. It consists of 16 convolutional and fully connected layers, organized into five blocks, with the convolutional layers using small 3x3 filters.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.09 |
| MR-V100 | 4.2.0 | 25.03 |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/vgg16-397923af.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
pip3 install -r ../../igie_common/requirements.txt
```

### Model Conversion

```bash
python3 ../../igie_common/export.py --model-name vgg16 --weight vgg16-397923af.pth --output vgg16.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
export RUN_DIR=../../igie_common/
```

### FP16

```bash
# Accuracy
bash scripts/infer_vgg16_fp16_accuracy.sh
# Performance
bash scripts/infer_vgg16_fp16_performance.sh
```

### INT8

```bash
# Accuracy
bash scripts/infer_vgg16_int8_accuracy.sh
# Performance
bash scripts/infer_vgg16_int8_performance.sh
```

## Model Results

| Model | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
| :----: | :----: | :----: | :----: | :----: | :----: |
| VGG16 | 32        | FP16      | 1830.53 | 71.55    | 90.37    |
| VGG16 | 32        | INT8      | 3528.01 | 71.53    | 90.32    |
