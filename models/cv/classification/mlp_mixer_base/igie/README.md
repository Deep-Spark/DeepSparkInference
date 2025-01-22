# MLP-Mixer Base

## Description

MLP-Mixer Base is a foundational model in the MLP-Mixer family, designed to use only MLP layers for vision tasks like image classification. Unlike CNNs and Vision Transformers, MLP-Mixer replaces both convolution and self-attention mechanisms with simple MLP layers to process spatial and channel-wise information independently.

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

Pretrained model: <https://download.openmmlab.com/mmclassification/v0/mlp-mixer/mixer-base-p16_3rdparty_64xb64_in1k_20211124-1377e3e0.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Model Conversion

```bash
# git clone mmpretrain
git clone -b v0.24.0 https://github.com/open-mmlab/mmpretrain.git

# export onnx model
python3 export.py --cfg mmpretrain/configs/mlp_mixer/mlp-mixer-base-p16_64xb64_in1k.py --weight mixer-base-p16_3rdparty_64xb64_in1k_20211124-1377e3e0.pth --output mlp_mixer_base.onnx

# Use onnxsim optimize onnx model
onnxsim mlp_mixer_base.onnx mlp_mixer_base_opt.onnx

```

## Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
```

### FP16

```bash
# Accuracy
bash scripts/infer_mlp_mixer_base_fp16_accuracy.sh
# Performance
bash scripts/infer_mlp_mixer_base_fp16_performance.sh
```

## Results

| Model           | BatchSize | Precision | FPS      | Top-1(%) | Top-5(%) |
| ----------------| --------- | --------- | -------- | -------- | -------- |
| MLP-Mixer-Base  | 32        | FP16      | 1477.15  | 72.545   | 90.035   |

## Reference

MLP-Mixer-Base: <https://github.com/open-mmlab/mmpretrain>
