# ShuffleNetV1

## Description
ShuffleNet V1 is a lightweight neural network architecture primarily used for image classification and object detection tasks. 
It uses techniques such as deep separable convolution and channel shuffle to reduce the number of parameters and computational complexity of the model, thereby achieving low computational resource consumption while maintaining high accuracy.

## Setup

### Install

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-dev

pip3 install tqdm
pip3 install tabulate
pip3 install onnx
pip3 install onnxsim
pip3 install opencv-python==4.6.0.66
pip3 install mmcls==0.24.0
pip3 install mmcv==1.5.3
```

### Download 

Dataset: https://www.image-net.org/download.php to download the validation dataset.

### Model Conversion 
```
mkdir checkpoints 
cd checkpoints
git clone -b v0.24.0 https://github.com/open-mmlab/mmpretrain.git
cd ..

python3 export_onnx.py   \
    --config_file ./checkpoints/mmpretrain/configs/shufflenet_v1/shufflenet-v1-1x_16xb64_in1k.py  \
    --checkpoint_file  https://download.openmmlab.com/mmclassification/v0/shufflenet_v1/shufflenet_v1_batch1024_imagenet_20200804-5d6cec73.pth \
    --output_model ./checkpoints/shufflenet_v1.onnx
```

## Inference
```
export PROJ_DIR=./
export DATASETS_DIR=/path/to/imagenet_val/
export CHECKPOINTS_DIR=./checkpoints
export RUN_DIR=./
export CONFIG_DIR=config/SHUFFLENET_V1_CONFIG

```
### FP16

```bash
# Accuracy
bash scripts/infer_shufflenet_v1_fp16_accuracy.sh
# Performance
bash scripts/infer_shufflenet_v1_fp16_performance.sh
```

## Results
Model        |BatchSize  |Precision |FPS      |Top-1(%)  |Top-5(%)
-------------|-----------|----------|---------|----------|--------
ShuffleNetV1 |    32     |   FP16   | 3619.89 |  66.17   | 86.54

