# Resnet101 (IxRT)

## Model Description

ResNet-101 is a variant of the ResNet (Residual Network) architecture, and it belongs to a family of deep neural networks introduced by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun in their 2016 paper, "Deep Residual Learning for Image Recognition." The ResNet architecture is known for its effective use of residual connections, which help in training very deep neural networks.

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

pip3 install -r reuirements.txt
```

### Model Conversion

```bash
mkdir checkpoints
python3 export_onnx.py --output_model checkpoints/resnet101.onnx
```

## Model Inference

```bash
export PROJ_DIR=./
export DATASETS_DIR=/path/to/imagenet_val/
export CHECKPOINTS_DIR=./checkpoints
export RUN_DIR=./
export CONFIG_DIR=config/RESNET101_CONFIG
```

### FP16

```bash
# Accuracy
bash scripts/infer_resnet101_fp16_accuracy.sh
# Performance
bash scripts/infer_resnet101_fp16_performance.sh
```

### INT8

```bash
# Accuracy
bash scripts/infer_resnet101_int8_accuracy.sh
# Performance
bash scripts/infer_resnet101_int8_performance.sh
```

## Model Results

| Model     | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
|-----------|-----------|-----------|---------|----------|----------|
| Resnet101 | 32        | FP16      | 2592.04 | 77.36    | 93.56    |
| Resnet101 | 32        | INT8      | 5760.69 | 76.88    | 93.43    |
