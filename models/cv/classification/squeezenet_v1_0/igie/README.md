# SqueezeNet1_0 (IGIE)

## Model Description

SqueezeNet1_0 is a lightweight convolutional neural network introduced in the paper "SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size." It was designed to achieve high classification accuracy with significantly fewer parameters, making it highly efficient for resource-constrained environments.The core innovation of SqueezeNet lies in the Fire Module, which reduces parameters using 1x1 convolutions in the "Squeeze layer" and expands feature maps through a mix of 1x1 and 3x3 convolutions in the "Expand layer." Additionally, delayed downsampling improves feature representation and accuracy.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.2.0     |  25.03  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/squeezenet1_0-b66bff10.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
pip3 install -r ../../igie_common/requirements.txt
```

### Model Conversion

```bash
python3 ../../igie_common/export.py --model-name squeezenet1_0 --weight squeezenet1_0-b66bff10.pth --output squeezenet1_0.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
export RUN_DIR=../../igie_common/
```

### FP16

```bash
# Accuracy
bash scripts/infer_squeezenet_v1_0_fp16_accuracy.sh
# Performance
bash scripts/infer_squeezenet_v1_0_fp16_performance.sh
```

## Model Results

| Model           | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
| :----: | :----: | :----: | :----: | :----: | :----: |
| Squeezenet_v1_0 | 32        | FP16      | 7777.50 | 58.08    | 80.39    |
