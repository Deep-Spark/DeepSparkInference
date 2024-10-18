# EfficientNet B2

## Description

EfficientNet B2 is a member of the EfficientNet family, a series of convolutional neural network architectures that are designed to achieve excellent accuracy and efficiency. Introduced by researchers at Google, EfficientNets utilize the compound scaling method, which uniformly scales the depth, width, and resolution of the network to improve accuracy and efficiency.

## Setup

### Install

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-dev

pip3 install tqdm
pip3 install onnx
pip3 install onnxsim
pip3 install tabulate
```

### Download

Pretrained model: <https://download.pytorch.org/models/efficientnet_b2_rwightman-c35c1473.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Model Conversion

```bash
python3 export.py --weight efficientnet_b2_rwightman-c35c1473.pth --output efficientnet_b2.onnx
```

## Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
```

### FP16

```bash
# Accuracy
bash scripts/infer_efficientnet_b1_fp16_accuracy.sh
# Performance
bash scripts/infer_efficientnet_b1_fp16_performance.sh
```

## Results

| Model           | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
| --------------- | --------- | --------- | ------- | -------- | -------- |
| EfficientNet_B2 | 32        | FP16      | 1450.04 | 77.79    | 93.76    |
