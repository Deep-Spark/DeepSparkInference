# DenseNet201

## Description

DenseNet201 is a deep convolutional neural network that stands out for its unique dense connection architecture, where each layer integrates features from all previous layers, effectively reusing features and reducing the number of parameters. This design not only enhances the network's information flow and parameter efficiency but also increases the model's regularization effect, helping to prevent overfitting. DenseNet201 consists of multiple dense blocks and transition layers, capable of capturing rich feature representations while maintaining computational efficiency, making it suitable for complex image recognition tasks.

## Setup

### Install

```bash
pip3 install -r requirements.txt
```

### Download

Pretrained model: <https://download.pytorch.org/models/densenet201-c1103571.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Model Conversion

```bash
python3 export.py --weight densenet201-c1103571.pth --output densenet201.onnx
```

## Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
```

### FP16

```bash
# Accuracy
bash scripts/infer_densenet201_fp16_accuracy.sh
# Performance
bash scripts/infer_densenet201_fp16_performance.sh
```

## Results

| Model       | BatchSize | Precision | FPS      | Top-1(%) | Top-5(%) |
| ----------- | --------- | --------- | -------- | -------- | -------- |
| DenseNet201 | 32        | FP16      | 758.592  | 76.851   | 93.338   |
