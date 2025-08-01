# CSPResNeXt50 (ixRT)

## Model Description

CSPResNeXt50 is a convolutional neural network that combines the CSPNet and ResNeXt architectures. It enhances computational efficiency and model performance through cross-stage partial connections and grouped convolutions, making it suitable for tasks such as image classification and object detection. This model improves learning capability and inference speed without significantly increasing the number of parameters.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0     |  25.09  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.openmmlab.com/mmclassification/v0/cspnet/cspresnext50_3rdparty_8xb32_in1k_20220329-2cc84d21.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
pip3 install -r ../../ixrt_common/requirements.txt
pip3 install mmcv==1.5.3 mmcls==0.24.0
```

### Model Conversion

```bash
# git clone mmpretrain
git clone -b v0.24.0 https://github.com/open-mmlab/mmpretrain.git

# export onnx model
python3 ../../ixrt_common/export_mmcls.py --cfg mmpretrain/configs/cspnet/cspresnext50_8xb32_in1k.py --weight cspresnext50_3rdparty_8xb32_in1k_20220329-2cc84d21.pth --output cspresnext50.onnx

# Use onnxsim optimize onnx model
mkdir -p checkpoints
onnxsim cspresnext50.onnx checkpoints/cspresnext50_sim.onnx
```

## Model Inference

```bash
export PROJ_DIR=./
export DATASETS_DIR=/path/to/imagenet_val/
export CHECKPOINTS_DIR=./checkpoints
export RUN_DIR=../../ixrt_common/
export CONFIG_DIR=../../ixrt_common/config/CSPResNeXt50_CONFIG
```

### FP16

```bash
# Accuracy
bash scripts/infer_cspresnext50_fp16_accuracy.sh
# Performance
bash scripts/infer_cspresnext50_fp16_performance.sh
```

## Model Results

| Model           | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
| --------------- | --------- | --------- | ------- | -------- | -------- |
| CSPResNeXt50 | 32        | FP16      | 827.76 | 80.04   | 94.94    |
