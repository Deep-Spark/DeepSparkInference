# DenseNet161

## Description

DenseNet161 is a convolutional neural network architecture that belongs to the family of Dense Convolutional Networks (DenseNets). Introduced as an extension to the previous DenseNet models, DenseNet161 offers improved performance and deeper network capacity, making it suitable for various computer vision tasks.

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
pip3 install cuda-python
```

### Download

Pretrained model: <https://download.pytorch.org/models/densenet161-8d451a50.pth>
Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Model Conversion

```bash
python3 export.py --weight densenet161-8d451a50.pth --output densenet161.onnx
```

## Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
```

### FP16

```bash
# Accuracy
bash scripts/infer_densenet_fp16_accuracy.sh
# Performance
bash scripts/infer_densenet_fp16_performance.sh
```

## Results

| Model       | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
| ----------- | --------- | --------- | ------- | -------- | -------- |
| DenseNet161 | 32        | FP16      | 589.784 | 0.7771   | 0.9354   |