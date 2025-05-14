# ResNext101_32x8d (IGIE)

## Model Description

ResNeXt101_32x8d is a deep convolutional neural network introduced in the paper "Aggregated Residual Transformations for Deep Neural Networks." It enhances the traditional ResNet architecture by incorporating group convolutions, offering a new dimension for scaling network capacity through "cardinality" (the number of groups) rather than merely increasing depth or width.The model consists of 101 layers and uses a configuration of 32 groups, each with a width of 8 channels. This design improves feature extraction while maintaining computational efficiency.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.2.0     |  25.03  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
pip3 install -r ../../igie_common/requirements.txt
```

### Model Conversion

```bash
python3 ../../igie_common/export.py --model-name resnext101_32x8d --weight resnext101_32x8d-8ba56ff5.pth --output resnext101_32x8d.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
export RUN_DIR=../../igie_common/
```

### FP16

```bash
# Accuracy
bash scripts/infer_resnext101_32x8d_fp16_accuracy.sh
# Performance
bash scripts/infer_resnext101_32x8d_fp16_performance.sh
```

## Model Results

| Model            | BatchSize | Precision | FPS    | Top-1(%) | Top-5(%) |
| ---------------- | --------- | --------- | ------ | -------- | -------- |
| ResNext101_32x8d | 32        | FP16      | 825.78 | 79.277   | 94.498   |
