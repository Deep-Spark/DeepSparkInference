# MNASNet0_75 (IGIE)

## Model Description

MNASNet0_75 is a lightweight convolutional neural network designed for mobile devices, introduced in the paper "MNASNet: Multi-Objective Neural Architecture Search for Mobile." The model leverages Multi-Objective Neural Architecture Search (NAS) to achieve a balance between accuracy and efficiency by optimizing both performance and computational cost. With a width multiplier of 0.75, MNASNet0_75 reduces the number of channels compared to the standard MNASNet (width multiplier of 1.0), resulting in fewer parameters.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.09 |
| MR-V100 | 4.2.0 | 25.03 |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/mnasnet0_75-7090bc5f.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
pip3 install -r ../../igie_common/requirements.txt
```

### Model Conversion

```bash
python3 ../../igie_common/export.py --model-name mnasnet0_75 --weight mnasnet0_75-7090bc5f.pth --output mnasnet0_75.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
export RUN_DIR=../../igie_common/
```

### FP16

```bash
# Accuracy
bash scripts/infer_mnasnet0_75_fp16_accuracy.sh
# Performance
bash scripts/infer_mnasnet0_75_fp16_performance.sh
```

## Model Results

| Model             | BatchSize | Precision | FPS      | Top-1(%) | Top-5(%) |
| ----------------- | --------- | --------- | -------- | -------- | -------- |
| MnasNet0_75       | 32        | FP16      | 6313.446 | 70.841   |  90.141  |
