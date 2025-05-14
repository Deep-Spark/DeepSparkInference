# DenseNet201 (IxRT)

## Model Description

DenseNet201 is a deep convolutional neural network that stands out for its unique dense connection architecture, where each layer integrates features from all previous layers, effectively reusing features and reducing the number of parameters. This design not only enhances the network's information flow and parameter efficiency but also increases the model's regularization effect, helping to prevent overfitting. DenseNet201 consists of multiple dense blocks and transition layers, capable of capturing rich feature representations while maintaining computational efficiency, making it suitable for complex image recognition tasks.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.2.0     |  25.03  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/densenet201-c1103571.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

pip3 install -r ../../ixrt_common/requirements.txt
```

### Model Conversion

```bash
mkdir checkpoints
python3 ../../ixrt_common/export.py --model-name densenet201 --weight densenet201-c1103571.pth --output checkpoints/densenet201.onnx
```

## Model Inference

```bash
export PROJ_DIR=./
export DATASETS_DIR=/path/to/imagenet_val/
export CHECKPOINTS_DIR=./checkpoints
export RUN_DIR=../../ixrt_common/
export CONFIG_DIR=../../ixrt_common/config/DENSENET201_CONFIG
```

### FP16

```bash
# Accuracy
bash scripts/infer_densenet201_fp16_accuracy.sh
# Performance
bash scripts/infer_densenet201_fp16_performance.sh
```

## Model Results

| Model       | BatchSize | Precision | FPS      | Top-1(%) | Top-5(%) |
| ----------- | --------- | --------- | -------- | -------- | -------- |
| DenseNet201 | 32        | FP16      | 788.946  | 76.88    | 93.37    |
