# DenseNet169

## Description

Dense Convolutional Network (DenseNet), connects each layer to every other layer in a feed-forward fashion. Whereas traditional convolutional networks with L layers have L connections - one between each layer and its subsequent layer - our network has L(L+1)/2 direct connections.

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

Pretrained model: <https://download.pytorch.org/models/densenet169-b2777c0a.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Model Conversion

```bash
python3 export.py --weight densenet169-b2777c0a.pth --output densenet169.onnx
cd data && mkdir checkpoints && cd checkpoints && mkdir densenet169
mv densenet169.onnx densenet169
```

## Inference


### FP16

```bash
cd deepsparkinference
# Accuracy
bash models/cv/classification/densenet169/ixrt/scripts/infer_densenet_fp16_accuracy.sh
# Performance
bash models/cv/classification/densenet169/ixrt/scripts/infer_densenet_fp16_performance.sh
```

## Results

| Model    | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
| -------- | --------- | --------- | ------- | -------- | -------- |
| DenseNet | 32        | FP16      | 1119.69 | 0.7558   | 0.9284   |
