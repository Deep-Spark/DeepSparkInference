# SqueezeNet 1.1

## Description
SqueezeNet 1.1 is a deep learning model for image classification, designed to be lightweight and efficient for deployment on resource-constrained devices.

It was developed by researchers at DeepScale and released in 2016.
## Setup

### Install
```
yum install mesa-libGL
pip3 install tqdm
pip3 install onnx
pip3 install onnxsim
pip3 install tabulate
pip3 install ppq
pip3 install pycuda
pip3 install opencv-python==4.6.0.66
```
### Download
Pretrained model: https://download.pytorch.org/models/squeezenet1_1-b8a52dc0.pth

Dataset: https://www.image-net.org/download.php to download the validation dataset.

### Model Conversion 
```
mkdir checkpoints 
python3 export_onnx.py --origin_model  /path/to/squeezenet1_1-b8a52dc0.pth --output_model checkpoints/squeezenetv11.onnx
```

## Inference
```
export PROJ_DIR=./
export DATASETS_DIR=/path/to/imagenet_val/
export CHECKPOINTS_DIR=./checkpoints
export RUN_DIR=./
export CONFIG_DIR=config/SQUEEZENET_V11_CONFIG

```
### FP16

```bash
# Accuracy
bash scripts/infer_squeezenet_v11_fp16_accuracy.sh
# Performance
bash scripts/infer_squeezenet_v11_fp16_performance.sh
```

### INT8

```bash
# Accuracy
bash scripts/infer_squeezenet_v11_int8_accuracy.sh
# Performance
bash scripts/infer_squeezenet_v11_int8_performance.sh
```

## Results 
Model          |BatchSize  |Precision |FPS      |Top-1(%)  |Top-5(%)
---------------|-----------|----------|---------|----------|--------
SqueezeNet 1.1 |           |   FP16   | 13701 |  0.58182  | 0.80622
SqueezeNet 1.1 |           |   INT8   | 20128 |  0.50966  | 0.77552 

