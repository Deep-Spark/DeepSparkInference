# GoogLeNet

## Description
GoogLeNet is a type of convolutional neural network based on the Inception architecture. It utilises Inception modules, which allow the network to choose between multiple convolutional filter sizes in each block. An Inception network stacks these modules on top of each other, with occasional max-pooling layers with stride 2 to halve the resolution of the grid.

## Setup

### Install
```bash
yum install mesa-libGL

pip3 install tqdm
pip3 install onnx
pip3 install onnxsim
pip3 install tabulate
```

### Download
Pretrained model: <https://download.pytorch.org/models/googlenet-1378be20.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Model Conversion
```bash
mkdir checkpoints
python3 export_onnx.py --origin_model /path/to/googlenet-1378be20.pth --output_model checkpoints/googlenet.onnx
```

## Inference
```bash
export PROJ_DIR=./
export DATASETS_DIR=/path/to/imagenet_val/
export CHECKPOINTS_DIR=./checkpoints
export RUN_DIR=./
export CONFIG_DIR=config/GOOGLENET_CONFIG
```
### FP16

```bash
# Accuracy
bash scripts/infer_googlenet_fp16_accuracy.sh
# Performance
bash scripts/infer_googlenet_fp16_performance.sh
```

### INT8
```bash
# Accuracy
bash scripts/infer_googlenet_int8_accuracy.sh
# Performance
bash scripts/infer_googlenet_int8_performance.sh
```

## Results

Model     |BatchSize  |Precision |FPS       |Top-1(%)  |Top-5(%)
----------|-----------|----------|----------|----------|--------
GoogLeNet |    32     |   FP16   | 6470.34  |  62.456  | 84.33
GoogLeNet |    32     |   INT8   | 9358.11  |  62.106  | 84.30

