# DenseNet169

## Model Description

Dense Convolutional Network (DenseNet), connects each layer to every other layer in a feed-forward fashion. Whereas traditional convolutional networks with L layers have L connections - one between each layer and its subsequent layer - our network has L(L+1)/2 direct connections.

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

Pretrained model: <https://download.pytorch.org/models/densenet169-b2777c0a.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Model Conversion

```bash
python3 export.py --weight densenet169-b2777c0a.pth --output densenet169.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
```

### FP16

```bash
# Accuracy
bash scripts/infer_densenet169_fp16_accuracy.sh
# Performance
bash scripts/infer_densenet169_fp16_performance.sh
```

## Model Results

| Model    | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
| -------- | --------- | --------- | ------- | -------- | -------- |
| DenseNet169 | 32        | FP16      | 1119.69 | 0.7558   | 0.9284   |
