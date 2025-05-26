# CSPResNet50 (IxRT)

## Model Description

Neural networks have enabled state-of-the-art approaches to achieve incredible results on computer vision tasks such as object detection.
CSPResNet50 is the one of best models.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.2.0     |  25.03  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.openmmlab.com/mmclassification/v0/cspnet/cspresnet50_3rdparty_8xb32_in1k_20220329-dd6dddfb.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

pip3 install -r ../../ixrt_common/requirements.txt

pip3 install mmcls==0.24.0 mmcv==1.5.3
```

### Model Conversion

```bash
mkdir checkpoints 
git clone -b v0.24.0 https://github.com/open-mmlab/mmpretrain.git

python3 export_onnx.py   \
    --cfg ./mmpretrain/configs/cspnet/cspresnet50_8xb32_in1k.py  \
    --weight cspresnet50_3rdparty_8xb32_in1k_20220329-dd6dddfb.pth \
    --output cspresnet50.onnx

onnxsim cspresnet50.onnx checkpoints/cspresnet50.onnx
```

## Model Inference

```bash
export PROJ_DIR=./
export DATASETS_DIR=/path/to/imagenet_val/
export CHECKPOINTS_DIR=./checkpoints
export RUN_DIR=../../ixrt_common/
export CONFIG_DIR=../../ixrt_common/config/CSPRESNET50_CONFIG
```

### FP16

```bash
# Accuracy
bash scripts/infer_cspresnet50_fp16_accuracy.sh
# Performance
bash scripts/infer_cspresnet50_fp16_performance.sh
```

### INT8

```bash
# Accuracy
bash scripts/infer_cspresnet50_int8_accuracy.sh
# Performance
bash scripts/infer_cspresnet50_int8_performance.sh
```

## Model Results

| Model       | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
| :----: | :----: | :----: | :----: | :----: | :----: |
| CSPResNet50 | 32        | FP16      | 4555.95 | 78.51    | 94.17    |
| CSPResNet50 | 32        | INT8      | 8801.94 | 78.15    | 93.95    |
