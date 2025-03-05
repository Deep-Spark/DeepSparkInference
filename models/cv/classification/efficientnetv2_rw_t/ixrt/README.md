# EfficientNetv2_rw_t (IGIE)

## Model Description

EfficientNetV2_rw_t is an enhanced version of the EfficientNet family of convolutional neural network architectures. It builds upon the success of its predecessors by introducing novel advancements aimed at further improving performance and efficiency in various computer vision tasks.

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

pip3 install -r requirements.txt
```

### Model Conversion

```bash
python3 export.py --weight efficientnetv2_t_agc-3620981a.pth --output efficientnetv2_rw_t.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
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
|---------------------|-----------|-----------|---------|----------|----------|
| Efficientnetv2_rw_t | 32        | FP16      | 1525.22 | 82.336   | 96.194   |
