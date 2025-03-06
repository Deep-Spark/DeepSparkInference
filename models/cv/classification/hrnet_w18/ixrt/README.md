# HRNet-W18 (IxRT)

## Model Description

HRNet-W18 is a powerful image classification model developed by Jingdong AI Research and released in 2020. It belongs to the HRNet (High-Resolution Network) family of models, known for their exceptional performance in various computer vision tasks.

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

pip3 install -r requirements.txt
```

### Model Conversion

```bash
mkdir checkpoints
python3 export_onnx.py --output_model checkpoints/hrnet-w18.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/path/to/imagenet_val/
export CHECKPOINTS_DIR=./checkpoints
export RUN_DIR=./
export CONFIG_DIR=config/HRNET_W18_CONFIG
```

### FP16

```bash
# Accuracy
bash scripts/infer_hrnet_w18_fp16_accuracy.sh
# Performance
bash scripts/infer_hrnet_w18_fp16_performance.sh
```

### INT8

```bash
# Accuracy
bash scripts/infer_hrnet_w18_int8_accuracy.sh
# Performance
bash scripts/infer_hrnet_w18_int8_performance.sh
```

## Model Results

| Model    | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
|----------|-----------|-----------|---------|----------|----------|
| ResNet50 | 32        | FP16      | 1474.26 | 0.76764  | 0.93446  |
| ResNet50 | 32        | INT8      | 1649.40 | 0.76158  | 0.93152  |
