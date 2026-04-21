# DeiT-Base (ixRT)

## Model Description

DeiT-Base (Data-efficient Image Transformer Base) is a vision transformer model that uses knowledge distillation to achieve competitive performance with fewer training resources.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.4.0 | 26.06 |

## Model Preparation

### Prepare Resources

Pretrained model: <http://files.deepspark.org.cn:880/deepspark/data/checkpoints/deit_b.onnx>

Download the [imagenet](https://www.image-net.org/download.php) to download the validation dataset.

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
mkdir checkpoints
cd checkpoints
wget http://files.deepspark.org.cn:880/deepspark/data/checkpoints/deit_b.onnx
```

## Model Inference

```bash
export PROJ_DIR=./
export DATASETS_DIR=/path/to/imagenet_val/
export CHECKPOINTS_DIR=./checkpoints
export RUN_DIR=../../ixrt_common/
export CONFIG_DIR=../../ixrt_common/config/DEIT_B_CONFIG
```

### FP16

```bash
# Test ACC
bash scripts/infer_deit_b_fp16_accuracy.sh
# Test FPS
bash scripts/infer_deit_b_fp16_performance.sh
```

## Model Results

| Model       | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
| ----------- | --------- | --------- | ------- | -------- | -------- |
| DeiT-Base   | 32        | FP16      | 596.381    | 81.7   | 95.6   |
