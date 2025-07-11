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
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

pip3 install -r ../../ixrt_common/requirements.txt
pip install timm
```

### Model Conversion

```bash
mkdir checkpoints
python3 ../../ixrt_common/export_timm.py --model-name efficientnetv2_rw_t --weight efficientnetv2_t_agc-3620981a.pth --output checkpoints/efficientnetv2_rw_t.onnx
```

## Model Inference

```bash
export PROJ_DIR=./
export DATASETS_DIR=/path/to/imagenet_val/
export CHECKPOINTS_DIR=./checkpoints
export RUN_DIR=../../ixrt_common/
export CONFIG_DIR=../../ixrt_common/config/EFFICIENTNETV2_RW_T_CONFIG
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
| Efficientnetv2_rw_t | 32        | FP16      | 1525.22 | 82.336   | 96.194   |
