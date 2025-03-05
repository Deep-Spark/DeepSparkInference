# ConvNeXt Small (IGIE)

## Model Description

The ConvNeXt Small model represents a significant stride in the evolution of convolutional neural networks (CNNs), introduced by researchers at Facebook AI Research (FAIR) and UC Berkeley. It is part of the ConvNeXt family, which challenges the dominance of Vision Transformers (ViTs) in the realm of visual recognition tasks.

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/convnext_small-0c510722.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
pip3 install -r requirements.txt
```

### Model Conversion

```bash
python3 export.py --weight convnext_small-0c510722.pth --output convnext_small.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
```

### FP16

```bash
# Accuracy
bash scripts/infer_convnext_small_fp16_accuracy.sh
# Performance
bash scripts/infer_convnext_small_fp16_performance.sh
```

## Model Results

| Model          | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
| -------------- | --------- | --------- | ------- | -------- | -------- |
| ConvNeXt Small | 32        | FP16      | 725.437 | 83.267   | 96.515   |
