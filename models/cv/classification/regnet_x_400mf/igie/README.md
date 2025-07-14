# RegNet_x_400mf (IGIE)

## Model Description

RegNet_x_400mf is a lightweight deep learning model designed with a regularized architecture, utilizing Bottleneck Blocks for efficient feature extraction. With lower computational complexity, it is well-suited for mid-scale image classification tasks in resource-constrained environments.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.2.0     |  25.06  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/regnet_x_400mf-adf1edd5.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
pip3 install -r ../../igie_common/requirements.txt
```

### Model Conversion

```bash
python3 ../../igie_common/export.py --model-name regnet_x_400mf --weight regnet_x_400mf-adf1edd5.pth --output regnet_x_400mf.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
export RUN_DIR=../../igie_common/
```

### FP16

```bash
# Accuracy
bash scripts/infer_regnet_x_400mf_fp16_accuracy.sh
# Performance
bash scripts/infer_regnet_x_400mf_fp16_performance.sh
```

## Model Results

| Model          | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
| :----: | :----: | :----: | :----: | :----: | :----: |
| RegNet_x_400mf | 32        | FP16      | 7951.452| 72.799   | 90.933   |
