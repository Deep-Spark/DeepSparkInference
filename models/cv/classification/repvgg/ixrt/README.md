# REPVGG

## Description
REPVGG is a family of convolutional neural network (CNN) architectures designed for image classification tasks.
It was developed by researchers at the University of Oxford and introduced in their paper titled "REPVGG: Making VGG-style ConvNets Great Again" in 2021.

## Setup

### Install 
```
yum install mesa-libGL
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
    --config_file ./checkpoints/mmpretrain/configs/repvgg/repvgg-A0_4xb64-coslr-120e_in1k.py \
    --checkpoint_file https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-A0_3rdparty_4xb64-coslr-120e_in1k_20210909-883ab98c.pth \
    --output_model ./checkpoints/repvgg_A0.onnx
```

## Inference
```
export PROJ_DIR=./
export DATASETS_DIR=/path/to/imagenet_val/
export CHECKPOINTS_DIR=./checkpoints
export RUN_DIR=./
export CONFIG_DIR=config/REPVGG_CONFIG

```
### FP16

```bash
# Accuracy
bash scripts/infer_repvgg_fp16_accuracy.sh
# Performance
bash scripts/infer_repvgg_fp16_performance.sh
```

## Results

Model  |BatchSize  |Precision |FPS      |Top-1(%)  |Top-5(%)
-------|-----------|----------|---------|----------|--------
REPVGG |    32     |   FP16   | 5725.37 |  72.41   | 90.49

