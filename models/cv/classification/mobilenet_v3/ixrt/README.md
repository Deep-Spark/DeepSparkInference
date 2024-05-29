# MobileNetV3

## Description
MobileNetV3 is a convolutional neural network that is tuned to mobile phone CPUs through a combination of hardware-aware network architecture search (NAS) complemented by the NetAdapt algorithm, and then subsequently improved through novel architecture advances. Advances include (1) complementary search techniques, (2) new efficient versions of nonlinearities practical for the mobile setting, (3) new efficient network design.

## Setup

### Install

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-dev

pip3 install tqdm
pip3 install onnx
pip3 install onnxsim
pip3 install tabulate
```

### Download
Pretrained model: <https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Model Conversion
```bash
mkdir checkpoints
python3 export_onnx.py --origin_model /path/to/mobilenet_v3_small-047dcff4.pth --output_model checkpoints/mobilenetv3.onnx
```

## Inference
```bash
export PROJ_DIR=./
export DATASETS_DIR=/path/to/imagenet_val/
export CHECKPOINTS_DIR=./checkpoints
export RUN_DIR=./
export CONFIG_DIR=config/MOBILENETV3_CONFIG
```
### FP16

```bash
# Accuracy
bash scripts/infer_mobilenetv3_fp16_accuracy.sh
# Performance
bash scripts/infer_mobilenetv3_fp16_performance.sh
```



## Results

Model       | BatchSize | Precision|   FPS    | Top-1(%) | Top-5(%)
------------|-----------|----------|----------|----------|--------
MobileNetV3 |    32     |   FP16   | 8464.36  |  67.62   | 87.42

