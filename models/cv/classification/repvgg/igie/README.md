# RepVGG

## Description

RepVGG is an innovative convolutional neural network architecture that combines the simplicity of VGG-style inference with a multi-branch topology during training. Through structural re-parameterization, RepVGG achieves high accuracy while significantly improving computational efficiency.

## Setup

### Install

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

pip3 install -r requirements.txt
```

### Download

Pretrained model: <https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-A0_8xb32_in1k_20221213-60ae8e23.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Model Conversion

```bash
# git clone mmpretrain
git clone -b v0.24.0 https://github.com/open-mmlab/mmpretrain.git

# export onnx model
python3 export.py --cfg mmpretrain/configs/repvgg/repvgg-A0_4xb64-coslr-120e_in1k.py --weight repvgg-A0_8xb32_in1k_20221213-60ae8e23.pth --output repvgg.onnx

```

## Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
```

### FP16

```bash
# Accuracy
bash scripts/infer_repvgg_fp16_accuracy.sh
# Performance
bash scripts/infer_repvgg_fp16_performance.sh
```

## Results

| Model  | BatchSize | Precision | FPS      | Top-1(%) | Top-5(%) |
| ------ | --------- | --------- | -------- | -------- | -------- |
| RepVGG | 32        | FP16      | 7423.035 | 72.345   | 90.543   |

## Reference

RepVGG: <https://github.com/open-mmlab/mmpretrain>
