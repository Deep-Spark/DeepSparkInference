# CSPDarkNet50

## Description

CSPDarkNet50 is an enhanced convolutional neural network architecture that reduces redundant computations by integrating cross-stage partial network features and truncating gradient flow, thereby maintaining high accuracy while lowering computational costs.

## Setup

### Install

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-dev

pip3 install onnx
pip3 install tqdm
pip3 install onnxsim
pip3 install mmcv==1.5.3
pip3 install mmcls
```

### Download

Pretrained model: <https://download.openmmlab.com/mmclassification/v0/cspnet/cspdarknet50_3rdparty_8xb32_in1k_20220329-bd275287.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Model Conversion

```bash
# git clone mmpretrain
git clone -b v0.24.0 https://github.com/open-mmlab/mmpretrain.git

# export onnx model
python3 export.py --cfg mmpretrain/configs/cspnet/cspdarknet50_8xb32_in1k.py --weight cspdarknet50_3rdparty_8xb32_in1k_20220329-bd275287.pth --output cspdarknet50.onnx

# Use onnxsim optimize onnx model
onnxsim cspdarknet50.onnx cspdarknet50_opt.onnx

```

## Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
```

### FP16

```bash
# Accuracy
bash scripts/infer_cspdarknet_fp16_accuracy.sh
# Performance
bash scripts/infer_cspdarknet_fp16_performance.sh
```

## Results

| Model        | BatchSize | Precision | FPS      | Top-1(%) | Top-5(%) |
| ------------ | --------- | --------- | -------- | -------- | -------- |
| CSPDarkNet50 | 32        | FP16      | 3214.387 | 79.063   | 94.492   |

## Reference

CSPDarkNet50: <https://github.com/open-mmlab/mmpretrain>
