# VGG16_BN (IGIE)

## Model Description

VGG16_BN is an improved version of VGG16, utilizing 3×3 small convolution kernels for feature extraction and adding Batch Normalization layers after each convolutional layer. This significantly enhances training stability and convergence speed. With a simple structure and excellent performance, it is widely used for image classification tasks.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.09 |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/vgg16_bn-6c64b313.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
pip3 install -r ../../igie_common/requirements.txt
```

### Model Conversion

```bash
python3 ../../igie_common/export.py --model-name vgg16_bn --weight vgg16_bn-6c64b313.pth --output vgg16_bn.onnx
```

## Model Inference

```bash
export PATH="/opt/sw_home/local/corex/bin:${PATH}"
export CUDA_PATH=/opt/sw_home/local/corex
export LD_LIBRARY_PATH="/usr/local/corex/lib64:/opt/sw_home/local/corex/lib64:/opt/sw_home/local/lib64:${LD_LIBRARY_PATH:-}"
export DATASETS_DIR=/Path/to/imagenet_val/
export RUN_DIR=../../igie_common/
```

### FP16

```bash
# Accuracy
bash scripts/infer_vgg16_bn_fp16_accuracy.sh
# Performance
bash scripts/infer_vgg16_bn_fp16_performance.sh
```

## Model Results

| Model    | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
| :----: | :----: | :----: | :----: | :----: | :----: |
| VGG16_BN | 32        | FP16      | 2157.70 | 73.343   | 91.477   |
