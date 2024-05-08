# HRNet-W18

## Description
HRNet-W18 is a powerful image classification model developed by Jingdong AI Research and released in 2020. It belongs to the HRNet (High-Resolution Network) family of models, known for their exceptional performance in various computer vision tasks.

## Setup

### Install
```bash
yum install mesa-libGL

pip3 install tqdm
pip3 install onnx
pip3 install onnxsim
pip3 install tabulate
pip3 install ppq
pip3 install mmpretrain
pip3 install mmcv-lite
```

### Download

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Model Conversion
```bash
mkdir checkpoints
python3 export_onnx.py --output_model checkpoints/hrnet-w18.onnx
```

## Inference
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

## Results

Model    |BatchSize  |Precision |FPS       |Top-1(%)  |Top-5(%)
---------|-----------|----------|----------|----------|--------
ResNet50 |           |          |          |          |      
ResNet50 |           |          |          |          |      


