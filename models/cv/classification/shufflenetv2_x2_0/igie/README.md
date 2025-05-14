# ShuffleNetV2_x2_0 (IGIE)

## Model Description

ShuffleNetV2_x2_0 is a lightweight convolutional neural network introduced in the paper "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design" by Megvii (Face++). It is designed to achieve high performance with low computational cost, making it ideal for mobile and embedded devices.The x2_0 in its name indicates a width multiplier of 2.0, meaning the model has twice as many channels compared to the baseline ShuffleNetV2_x1_0. It employs Channel Shuffle to enable efficient information exchange between grouped convolutions, addressing the limitations of group convolutions. The core building block, the ShuffleNetV2 block, features a split-merge design and channel shuffle mechanism, ensuring both high efficiency and accuracy.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.2.0     |  25.03  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/shufflenetv2_x2_0-8be3c8ee.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
pip3 install -r requirements.txt
```

### Model Conversion

```bash
python3 export.py --weight shufflenetv2_x2_0-8be3c8ee.pth --output shufflenetv2_x2_0.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
```

### FP16

```bash
# Accuracy
bash scripts/infer_shufflenetv2_x2_0_fp16_accuracy.sh
# Performance
bash scripts/infer_shufflenetv2_x2_0_fp16_performance.sh
```

## Model Results

| Model             | BatchSize | Precision | FPS      | Top-1(%) | Top-5(%) |
| ----------------- | --------- | --------- | -------- | -------- | -------- |
| ShuffleNetV2_x2_0 | 32        | FP16      | 5439.098 | 76.176   | 92.860   |
