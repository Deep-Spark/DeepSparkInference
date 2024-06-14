# Res2Net50

## Description

A novel building block for CNNs, namely Res2Net, by constructing hierarchical residual-like connections within one single residual block. The Res2Net represents multi-scale features at a granular level and increases the range of receptive fields for each network layer. The proposed Res2Net block can be plugged into the state-of-the-art backbone CNN models, e.g., ResNet, ResNeXt, and DLA. We evaluate the Res2Net block on all these models and demonstrate consistent performance gains over baseline models on widely-used datasets, e.g., CIFAR-100 and ImageNet. Further ablation studies and experimental results on representative computer vision tasks, i.e., object detection, class activation mapping, and salient object detection, further verify the superiority of the Res2Net over the state-of-the-art baseline methods.

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

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Model Conversion

```bash
mkdir checkpoints
python3 export_onnx.py --output_model checkpoints/res2net50.onnx
```

## Inference

```bash
export PROJ_DIR=./
export DATASETS_DIR=/path/to/imagenet_val/
export CHECKPOINTS_DIR=./checkpoints
export RUN_DIR=./
export CONFIG_DIR=config/RES2NET50_CONFIG
```

### FP16

```bash
# Accuracy
bash scripts/infer_res2net50_fp16_accuracy.sh
# Performance
bash scripts/infer_res2net50_fp16_performance.sh
```

### INT8

```bash
# Accuracy
bash scripts/infer_res2net50_int8_accuracy.sh
# Performance
bash scripts/infer_res2net50_int8_performance.sh
```

## Results

Model     |BatchSize  |Precision |FPS       |Top-1(%) |Top-5(%)
----------|-----------|----------|----------|---------|--------
Res2Net50 |    32     |   FP16   | 921.37   |  77.92  | 93.71
Res2Net50 |    32     |   INT8   | 1933.74  |  77.80  | 93.62
