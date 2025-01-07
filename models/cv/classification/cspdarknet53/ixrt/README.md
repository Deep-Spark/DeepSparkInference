# CSPDarkNet53

## Description

CSPDarkNet53 is an enhanced convolutional neural network architecture that reduces redundant computations by integrating cross-stage partial network features and truncating gradient flow, thereby maintaining high accuracy while lowering computational costs.

## Setup

### Install

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-dev

pip3 install -r requirements.txt
```

### Download

Pretrained model: <https://download.openmmlab.com/mmclassification/v0/cspnet/cspdarknet50_3rdparty_8xb32_in1k_20220329-bd275287.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Model Conversion

```bash
# git clone mmpretrain
git clone -b v0.24.0 https://github.com/open-mmlab/mmpretrain.git

# export onnx model
## cspdarknet50 is actually cspdarknet53
wget -O cspdarknet53_3rdparty_8xb32_in1k_20220329-bd275287.pth https://download.openmmlab.com/mmclassification/v0/cspnet/cspdarknet50_3rdparty_8xb32_in1k_20220329-bd275287.pth

python3 export.py --cfg mmpretrain/configs/cspnet/cspdarknet50_8xb32_in1k.py --weight cspdarknet53_3rdparty_8xb32_in1k_20220329-bd275287.pth --output cspdarknet53.onnx

# Use onnxsim optimize onnx model
mkdir -p checkpoints
onnxsim cspdarknet5.onnx checkpoints/cspdarknet53_sim.onnx

```

## Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
export CHECKPOINTS_DIR=/Path/to/checkpoints/
export CONFIG_DIR=./config/CSPDARKNET53_CONFIG
```

### FP16

```bash
# Accuracy
bash scripts/infer_cspdarknet53_fp16_accuracy.sh 
# Performance
bash scripts/infer_cspdarknet53_fp16_performance.sh
```

### INT8

```bash
# Accuracy
bash scripts/infer_cspdarknet53_int8_accuracy.sh 
# Performance
bash scripts/infer_cspdarknet53_int8_performance.sh
```

## Results

| Model        | BatchSize | Precision | FPS      | Top-1(%) | Top-5(%) |
| ------------ | --------- | --------- | -------- | -------- | -------- |
| CSPDarkNet53 | 32        | FP16      | 3282.318 | 79.09    | 94.52    |
| CSPDarkNet53 | 32        | INT8      | 6335.86  | 75.49    | 92.66    |

## Reference

CSPDarkNet53: <https://github.com/open-mmlab/mmpretrain>
