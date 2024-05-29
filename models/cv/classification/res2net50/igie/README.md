# Res2Net50

## Description
Res2Net50 is a convolutional neural network architecture that introduces the concept of "Residual-Residual Networks" (Res2Nets) to enhance feature representation and model expressiveness, particularly in image recognition tasks.The key innovation of Res2Net50 lies in its hierarchical feature aggregation mechanism, which enables the network to capture multi-scale features more effectively. Unlike traditional ResNet architectures, Res2Net50 incorporates multiple parallel pathways within each residual block, allowing the network to dynamically adjust the receptive field size and aggregate features across different scales.

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

Pretrained model: <https://download.openmmlab.com/mmclassification/v0/res2net/res2net50-w14-s8_3rdparty_8xb32_in1k_20210927-bc967bf1.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Model Conversion
```bash
# git clone mmpretrain
git clone -b v0.24.0 https://github.com/open-mmlab/mmpretrain.git

# export onnx model
python3 export.py --cfg mmpretrain/configs/res2net/res2net50-w14-s8_8xb32_in1k.py --weight res2net50-w14-s8_3rdparty_8xb32_in1k_20210927-bc967bf1.pth --output res2net50.onnx

# Use onnxsim optimize onnx model
onnxsim res2net50.onnx res2net50_opt.onnx

```

## Inference
```bash
export DATASETS_DIR=/Path/to/imagenet_val/
```
### FP16

```bash
# Accuracy
bash scripts/infer_res2net50_fp16_accuracy.sh
# Performance
bash scripts/infer_res2net50_fp16_performance.sh
```

## Results

Model     |BatchSize  |Precision |FPS       |Top-1(%)  |Top-5(%)
----------|-----------|----------|----------|----------|--------
Res2Net50 |    32     |   FP16   | 1641.961 |  78.139  | 93.826

## Reference

Res2Net50: https://github.com/open-mmlab/mmpretrain
