# EfficientNetv2_rw_t (IGIE)

## Model Description

EfficientNetV2_rw_t is an enhanced version of the EfficientNet family of convolutional neural network architectures. It builds upon the success of its predecessors by introducing novel advancements aimed at further improving performance and efficiency in various computer vision tasks.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.09 |
| MR-V100 | 4.2.0 | 25.03 |

## Model Preparation

### Prepare Resources

Pretrained model: <https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnetv2_t_agc-3620981a.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
pip3 install -r ../../igie_common/requirements.txt
pip3 install timm
```

### Model Conversion

```bash
python3 ../../igie_common/export_timm.py --model-name efficientnetv2_rw_t --weight efficientnetv2_t_agc-3620981a.pth --output efficientnetv2_rw_t.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
export RUN_DIR=../../igie_common/
```

### FP16

```bash
# Accuracy
bash scripts/infer_efficientnetv2_rw_t_fp16_accuracy.sh
# Performance
bash scripts/infer_efficientnetv2_rw_t_fp16_performance.sh
```

## Model Results

| Model               | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
| :----: | :----: | :----: | :----: | :----: | :----: |
| Efficientnetv2_rw_t | 32        | FP16      | 831.678 | 82.306   | 96.163   |
