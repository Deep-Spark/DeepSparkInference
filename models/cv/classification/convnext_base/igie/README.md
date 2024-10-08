# ConvNext Base

## Description

The ConvNeXt Base model represents a significant stride in the evolution of convolutional neural networks (CNNs), introduced by researchers at Facebook AI Research (FAIR) and UC Berkeley. It is part of the ConvNeXt family, which challenges the dominance of Vision Transformers (ViTs) in the realm of visual recognition tasks.

## Setup

### Install

```bash
pip3 install onnx
pip3 install tqdm
```

### Download

Pretrained model: <https://download.pytorch.org/models/convnext_base-6075fbad.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Model Conversion

```bash
python3 export.py --weight convnext_base-6075fbad.pth --output convnext_base.onnx
```

## Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
```

### FP16

```bash
# Accuracy
bash scripts/infer_convnext_base_fp16_accuracy.sh
# Performance
bash scripts/infer_convnext_base_fp16_performance.sh
```

## Results

| Model          | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
| -------------- | --------- | --------- | ------- | -------- | -------- |
| ConvNext Base  | 32        | FP16      | 589.669 | 83.661   | 96.699   |
