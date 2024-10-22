# RegNet_y_1_6gf

## Description

RegNet is a family of models designed for image classification tasks, as described in the paper "Designing Network Design Spaces". The RegNet design space provides simple and fast networks that work well across a wide range of computational budgets.The architecture of RegNet models is based on the principle of designing network design spaces, which allows for a more systematic exploration of possible network architectures. This makes it easier to understand and modify the architecture.RegNet_y_1_6gf is a specific model within the RegNet family, designed for image classification tasks.

## Setup

### Install

```bash
pip3 install -r requirements.txt
```

### Download

Pretrained model: <https://download.pytorch.org/models/regnet_y_1_6gf-b11a554e.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Model Conversion

```bash
python3 export.py --weight regnet_y_1_6gf-b11a554e.pth --output regnet_y_1_6gf.onnx
```

## Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
```

### FP16

```bash
# Accuracy
bash scripts/infer_regnet_y_1_6gf_fp16_accuracy.sh
# Performance
bash scripts/infer_regnet_y_1_6gf_fp16_performance.sh
```

## Results

| Model          | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
| -------------- | --------- | --------- | ------- | -------- | -------- |
| RegNet_y_1_6gf | 32        | FP16      | 1785.44 | 77.933   | 93.948   |
