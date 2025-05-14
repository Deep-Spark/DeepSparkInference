# ShuffleNetV2_x1_5 (IGIE)

## Model Description

ShuffleNetV2_x1_5 is a lightweight convolutional neural network specifically designed for efficient image recognition tasks on resource-constrained devices. It achieves high performance and low latency through the introduction of channel shuffling and pointwise group convolutions. Despite its small model size, it offers high accuracy and is suitable for a variety of vision tasks in mobile devices and embedded systems.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.2.0     |  25.03  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/shufflenetv2_x1_5-3c479a10.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
pip3 install -r ../../igie_common/requirements.txt
```

### Model Conversion

```bash
python3 ../../igie_common/export.py --model-name shufflenet_v2_x1_5 --weight shufflenetv2_x1_5-3c479a10.pth --output shufflenetv2_x1_5.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
export RUN_DIR=../../igie_common/
```

### FP16

```bash
# Accuracy
bash scripts/infer_shufflenetv2_x1_5_fp16_accuracy.sh
# Performance
bash scripts/infer_shufflenetv2_x1_5_fp16_performance.sh
```

## Model Results

| Model             | BatchSize | Precision | FPS      | Top-1(%) | Top-5(%) |
| ----------------- | --------- | --------- | -------- | -------- | -------- |
| ShuffleNetV2_x1_5 | 32        | FP16      | 7478.728 | 72.755   | 91.031   |
