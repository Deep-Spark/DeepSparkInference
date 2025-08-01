# Wide ResNet50 (ixRT)

## Model Description

The distinguishing feature of Wide ResNet50 lies in its widened architecture compared to traditional ResNet models. By increasing the width of the residual blocks, Wide ResNet50 enhances the capacity of the network to capture richer and more diverse feature representations, leading to improved performance on various visual recognition tasks.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.09 |
| MR-V100 | 4.2.0 | 25.03 |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

pip3 install -r ../../ixrt_common/requirements.txt
```

### Model Conversion

```bash
mkdir -p checkpoints/
python3 ../../ixrt_common/export.py --model-name wide_resnet50_2 --weight wide_resnet50_2-95faca4d.pth --output checkpoints/wide_resnet50.onnx
```

## Model Inference

```bash
export PROJ_DIR=./
export DATASETS_DIR=/path/to/imagenet_val/
export CHECKPOINTS_DIR=./checkpoints
export RUN_DIR=../../ixrt_common/
export CONFIG_DIR=../../ixrt_common/config/WIDE_RESNET50_CONFIG
```

### FP16

```bash
# Accuracy
bash scripts/infer_wide_resnet50_fp16_accuracy.sh
# Performance
bash scripts/infer_wide_resnet50_fp16_performance.sh
```

### INT8

```bash
# Accuracy
bash scripts/infer_wide_resnet50_int8_accuracy.sh
# Performance
bash scripts/infer_wide_resnet50_int8_performance.sh
```

## Model Results

| Model         | BatchSize | Precision | FPS      | Top-1(%) | Top-5(%) |
| ------------- | --------- | --------- | -------- | -------- | -------- |
| Wide ResNet50 | 32        | FP16      | 2478.551 | 78.486   | 94.084   |
| Wide ResNet50 | 32        | INT8      | 5981.702 | 76.956   | 93.920   |
