# EfficientnetV2

## Description
EfficientNetV2 is an improved version of the EfficientNet architecture proposed by Google, aiming to enhance model performance and efficiency. Unlike the original EfficientNet, EfficientNetV2 features a simplified design and incorporates a series of enhancement strategies to further boost performance.

## Setup

### Install
```bash
yum install mesa-libGL
pip3 install tqdm
pip3 install onnx
pip3 install onnxsim
pip3 install tabulate
pip3 install timm
pip3 install ppq
pip3 install protobuf==3.20.0
```

### Download
Pretrained model: <https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnetv2_t_agc-3620981a.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Model Conversion
```bash
mkdir checkpoints
git clone https://github.com/huggingface/pytorch-image-models.git
cp /Path/to/ixrt/export_onnx.py pytorch-image-models/timm/models
cd pytorch-image-models/timm/models
rm _builder.py
mv /Path/ixrt/_builder.py pytorch-image-models/timm/models
cd pytorch-image-models/timm
mkdir -p /root/.cache/torch/hub/checkpoints/
wget -P /root/.cache/torch/hub/checkpoints/ https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnetv2_t_agc-3620981a.pth
python3 -m models.export_onnx --output_model checkpoints/efficientnet.onnx
```

## Inference
```bash
export PROJ_DIR=/Path/to/efficientnet_v2/ixrt
export DATASETS_DIR=/path/to/imagenet_val/
export CHECKPOINTS_DIR=./checkpoints
export RUN_DIR=/Path/to/efficientnet_v2/ixrt
export CONFIG_DIR=/Path/to/config/EFFICIENTNET_V2T_CONFIG
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```
### FP16

```bash
# Accuracy
bash scripts/infer_efficientnet_fp16_accuracy.sh
# Performance
bash scripts/infer_efficientnet_fp16_performance.sh
```

### INT8
```bash
# Accuracy
bash scripts/infer_efficientnet_int8_accuracy.sh
# Performance
bash scripts/infer_efficientnet_int8_performance.sh
```



## Results

Model          | BatchSize | Precision |   FPS    | Top-1(%) | Top-5(%)
---------------|-----------|-----------|----------|----------|--------
EfficientnetV2 |    32     |   FP16    | 1882.87  |  82.14   | 96.16
EfficientnetV2 |    32     |   INT8    | 2595.96  |  81.50   | 95.96
