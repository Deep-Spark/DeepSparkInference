# RepVGG

## Model Description

REPVGG is a family of convolutional neural network (CNN) architectures designed for image classification tasks.
It was developed by researchers at the University of Oxford and introduced in their paper titled "REPVGG: Making VGG-style ConvNets Great Again" in 2021.

## Model Preparation

### Install Dependencies

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

pip3 install -r requirements.txt
```

### Prepare Resources

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Model Conversion

```bash
mkdir checkpoints 
git clone -b v0.24.0 https://github.com/open-mmlab/mmpretrain.git

python3 export_onnx.py   \
    --config_file ./mmpretrain/configs/repvgg/repvgg-A0_4xb64-coslr-120e_in1k.py \
    --checkpoint_file https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-A0_3rdparty_4xb64-coslr-120e_in1k_20210909-883ab98c.pth \
    --output_model ./checkpoints/repvgg_A0.onnx
```

## Model Inference

```bash
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

## Model Results

| Model  | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
| ------ | --------- | --------- | ------- | -------- | -------- |
| RepVGG | 32        | FP16      | 5725.37 | 72.41    | 90.49    |
