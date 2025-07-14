# VGG13_BN (IGIE)

## Model Description

VGG13_BN is an improved version of VGG13, utilizing 3×3 small convolution kernels for feature extraction and adding Batch Normalization layers after each convolutional and fully connected layer. This significantly enhances training stability and convergence speed. With a simple structure and excellent performance, it is ideal for image classification tasks but requires substantial computational resources.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0     |  25.09  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/vgg13_bn-abd245e5.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
pip3 install -r ../../igie_common/requirements.txt
```

### Model Conversion

```bash
python3 ../../igie_common/export.py --model-name vgg13_bn --weight vgg13_bn-abd245e5.pth --output vgg13_bn.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
export RUN_DIR=../../igie_common/
```

### FP16

```bash
# Accuracy
bash scripts/infer_vgg13_bn_fp16_accuracy.sh
# Performance
bash scripts/infer_vgg13_bn_fp16_performance.sh
```

## Model Results

| Model     | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
| :-------: | :-------: | :-------: | :-----: | :------: | :------: |
| VGG13_BN  | 32        | FP16      | 2597.53 | 71.539   | 90.347   |
