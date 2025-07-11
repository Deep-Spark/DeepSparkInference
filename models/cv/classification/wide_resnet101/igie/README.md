# Wide ResNet101 (IGIE)

## Model Description

Wide ResNet101 is a variant of the ResNet architecture that focuses on increasing the network's width (number of channels per layer) rather than its depth. This approach, inspired by the paper "Wide Residual Networks," balances model depth and width to achieve better performance while avoiding the drawbacks of overly deep networks, such as vanishing gradients and feature redundancy.Wide ResNet101 builds upon the standard ResNet101 architecture but doubles (or quadruples) the number of channels in each residual block. This results in significantly improved feature representation, making it suitable for complex tasks like image classification, object detection, and segmentation.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.09 |
| MR-V100 | 4.2.0 | 25.03 |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
pip3 install -r ../../igie_common/requirements.txt
```

### Model Conversion

```bash
python3 ../../igie_common/export.py --model-name wide_resnet101_2 --weight wide_resnet101_2-32ee1156.pth --output wide_resnet101.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
export RUN_DIR=../../igie_common/
```

### FP16

```bash
# Accuracy
bash scripts/infer_wide_resnet101_fp16_accuracy.sh
# Performance
bash scripts/infer_wide_resnet101_fp16_performance.sh
```

## Model Results

| Model          | BatchSize | Precision | FPS      | Top-1(%) | Top-5(%) |
| -------------- | --------- | --------- | -------- | -------- | -------- |
| Wide ResNet101 | 32        | FP16      | 1339.037 | 78.459   | 94.052   |
