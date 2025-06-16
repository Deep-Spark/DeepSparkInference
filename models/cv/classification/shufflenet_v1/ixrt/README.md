# ShuffleNetV1 (IxRT)

## Model Description

ShuffleNet V1 is a lightweight neural network architecture primarily used for image classification and object detection tasks.
It uses techniques such as deep separable convolution and channel shuffle to reduce the number of parameters and computational complexity of the model, thereby achieving low computational resource consumption while maintaining high accuracy.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.2.0     |  25.03  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.openmmlab.com/mmclassification/v0/shufflenet_v1/shufflenet_v1_batch1024_imagenet_20200804-5d6cec73.pth>

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
    --cfg ./mmpretrain/configs/shufflenet_v1/shufflenet-v1-1x_16xb64_in1k.py  \
    --weight  ./shufflenet_v1_batch1024_imagenet_20200804-5d6cec73.pth \
    --output ./checkpoints/shufflenetv1.onnx
```

## Model Inference

```bash
export PROJ_DIR=./
export DATASETS_DIR=/path/to/imagenet_val/
export CHECKPOINTS_DIR=./checkpoints
export RUN_DIR=../../ixrt_common/
export CONFIG_DIR=../../ixrt_common/config/SHUFFLENET_V1_CONFIG
```

### FP16

```bash
# Accuracy
bash scripts/infer_shufflenet_v1_fp16_accuracy.sh
# Performance
bash scripts/infer_shufflenet_v1_fp16_performance.sh
```

## Model Results

| Model        | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
| :----: | :----: | :----: | :----: | :----: | :----: |
| ShuffleNetV1 | 32        | FP16      | 3619.89 | 66.17    | 86.54    |
