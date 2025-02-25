# CSPResNet50

## Description

CSPResNet50 combines the strengths of ResNet50 and CSPNet (Cross-Stage Partial Network) to create a more efficient and high-performing architecture. By splitting and fusing feature maps across stages, CSPResNet50 reduces redundant computations, optimizes gradient flow, and enhances feature representation.

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

Pretrained model: <https://download.openmmlab.com/mmclassification/v0/cspnet/cspresnet50_3rdparty_8xb32_in1k_20220329-dd6dddfb.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Model Conversion

```bash
# git clone mmpretrain
git clone -b v0.24.0 https://github.com/open-mmlab/mmpretrain.git

# export onnx model
python3 export.py --cfg mmpretrain/configs/cspnet/cspresnet50_8xb32_in1k.py --weight cspresnet50_3rdparty_8xb32_in1k_20220329-dd6dddfb.pth --output cspresnet50.onnx

# Use onnxsim optimize onnx model
onnxsim cspresnet50.onnx cspresnet50_opt.onnx

```

## Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
```

### FP16

```bash
# Accuracy
bash scripts/infer_cspresnet50_fp16_accuracy.sh
# Performance
bash scripts/infer_cspresnet50_fp16_performance.sh
```

## Results

| Model        | BatchSize | Precision | FPS      | Top-1(%) | Top-5(%) |
| ------------ | --------- | --------- | -------- | -------- | -------- |
| CSPResNet50  | 32        | FP16      | 4553.80  | 78.507   | 94.142   |

## Reference

CSPResNet50: <https://github.com/open-mmlab/mmpretrain>
