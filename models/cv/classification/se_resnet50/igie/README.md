# SEResNet50

## Description

SEResNet50 is an enhanced version of the ResNet50 network integrated with Squeeze-and-Excitation (SE) blocks, which strengthens the network's feature expression capability by explicitly emphasizing useful features and suppressing irrelevant ones. This improvement enables SEResNet50 to demonstrate higher accuracy in various visual recognition tasks compared to the standard ResNet50.

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

Pretrained model: <https://download.openmmlab.com/mmclassification/v0/se-resnet/se-resnet50_batch256_imagenet_20200804-ae206104.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Model Conversion

```bash
# git clone mmpretrain
git clone -b v0.24.0 https://github.com/open-mmlab/mmpretrain.git

# export onnx model
python3 export.py --cfg mmpretrain/configs/seresnet/seresnet50_8xb32_in1k.py --weight se-resnet50_batch256_imagenet_20200804-ae206104.pth --output seresnet50.onnx

```

## Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
```

### FP16

```bash
# Accuracy
bash scripts/infer_seresnet_fp16_accuracy.sh
# Performance
bash scripts/infer_seresnet_fp16_performance.sh
```

## Results

| Model      | BatchSize | Precision | FPS      | Top-1(%) | Top-5(%) |
| ---------- | --------- | --------- | -------- | -------- | -------- |
| SEResNet50 | 32        | FP16      | 2548.268 | 77.709   | 93.812   |

## Reference

SE_ResNet50: <https://github.com/open-mmlab/mmpretrain>
