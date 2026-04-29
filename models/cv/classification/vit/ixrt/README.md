# Vision Transformer (ViT) (ixRT)

## Model Description

Vision Transformer (ViT) applies a pure transformer to images without any convolution. It divides an image into patches and processes them through transformer layers.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.4.0 | 26.06 |

## Model Preparation

### Prepare Resources

Pretrained model: <http://files.deepspark.org.cn:880/deepspark/data/checkpoints/vit_b_16_sim.onnx>

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
wget http://files.deepspark.org.cn:880/deepspark/data/checkpoints/vit_b_16_sim.onnx
```

## Model Inference

```bash
export PROJ_DIR=./
export DATASETS_DIR=/path/to/imagenet_val/
export CHECKPOINTS_DIR=./checkpoints
export RUN_DIR=../../ixrt_common/
export CONFIG_DIR=../../ixrt_common/config/VIT_CONFIG
```

### FP16

```bash
# Test ACC
bash scripts/infer_vit_fp16_accuracy.sh
# Test FPS
bash scripts/infer_vit_fp16_performance.sh
```

## Model Results

| Model       | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
| ----------- | --------- | --------- | ------- | -------- | -------- |
| ViT-B/16    | 32        | FP16      | 461.038  | 81.1    | 95.3  |