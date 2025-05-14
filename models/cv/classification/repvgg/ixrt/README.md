# RepVGG (IxRT)

## Model Description

REPVGG is a family of convolutional neural network (CNN) architectures designed for image classification tasks.
It was developed by researchers at the University of Oxford and introduced in their paper titled "REPVGG: Making VGG-style ConvNets Great Again" in 2021.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.2.0     |  25.03  |

## Model Preparation

### Prepare Resources

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

python3 ../../ixrt_common/export_mmcls.py   \
    --cfg ./mmpretrain/configs/repvgg/repvgg-A0_4xb64-coslr-120e_in1k.py \
    --weight repvgg-A0_3rdparty_4xb64-coslr-120e_in1k_20210909-883ab98c.pth \
    --output repvgg_A0.onnx

onnxsim repvgg_A0.onnx checkpoints/repvgg_A0.onnx
```

## Model Inference

```bash
export PROJ_DIR=./
export DATASETS_DIR=/path/to/imagenet_val/
export CHECKPOINTS_DIR=./checkpoints
export RUN_DIR=../../ixrt_common/
export CONFIG_DIR=../../ixrt_common/config/REPVGG_CONFIG
```

### FP16

```bash
# Accuracy
bash scripts/infer_repvgg_fp16_accuracy.sh
# Performance
bash scripts/infer_repvgg_fp16_performance.sh
```

## Model Results

| Model  | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
| ------ | --------- | --------- | ------- | -------- | -------- |
| RepVGG | 32        | FP16      | 5725.37 | 72.41    | 90.49    |
