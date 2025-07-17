# MNASNet1_3 (IGIE)

## Model Description

MNASNet1_3 is a lightweight deep learning model optimized through neural architecture search (NAS). It uses Inverted Residual Blocks and a width multiplier of 1.3, balancing efficiency and accuracy. With a simple structure and excellent performance, it is particularly suited for mobile devices and resource-constrained environments.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0     |  25.09  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/mnasnet1_3-a4c69d6f.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
pip3 install -r ../../igie_common/requirements.txt
```

### Model Conversion

```bash
python3 ../../igie_common/export.py --model-name mnasnet1_3 --weight mnasnet1_3-a4c69d6f.pth --output mnasnet1_3.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
export RUN_DIR=../../igie_common/
```

### FP16

```bash
# Accuracy
bash scripts/infer_mnasnet1_3_fp16_accuracy.sh
# Performance
bash scripts/infer_mnasnet1_3_fp16_performance.sh
```

## Model Results

| Model             | BatchSize | Precision | FPS      | Top-1(%) | Top-5(%) |
| ----------------- | --------- | --------- | -------- | -------- | -------- |
| MnasNet1_3        | 32        | FP16      | 4282.213 | 76.054   |  93.244  |
