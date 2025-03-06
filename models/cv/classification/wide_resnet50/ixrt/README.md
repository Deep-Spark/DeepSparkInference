# Wide ResNet50 (IxRT)

## Model Description

The distinguishing feature of Wide ResNet50 lies in its widened architecture compared to traditional ResNet models. By increasing the width of the residual blocks, Wide ResNet50 enhances the capacity of the network to capture richer and more diverse feature representations, leading to improved performance on various visual recognition tasks.

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
pip3 install -r requirements.txt
```

### Model Conversion

```bash
mkdir -p checkpoints/
python3 export.py --weight wide_resnet50_2-95faca4d.pth --output checkpoints/wide_resnet50.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
export CHECKPOINTS_DIR=./checkpoints
export RUN_DIR=./
export CONFIG_DIR=config/WIDE_RESNET50_CONFIG
```

### FP16

```bash
# Accuracy
bash scripts/infer_wide_resnet50_fp16_accuracy.sh
# Performance
bash scripts/infer_wide_resnet50_fp16_performance.sh
```

### INT8

```bash
# Accuracy
bash scripts/infer_wide_resnet50_int8_accuracy.sh
# Performance
bash scripts/infer_wide_resnet50_int8_performance.sh
```

## Model Results

| Model         | BatchSize | Precision | FPS      | Top-1(%) | Top-5(%) |
| ------------- | --------- | --------- | -------- | -------- | -------- |
| Wide ResNet50 | 32        | FP16      | 2478.551 | 78.486   | 94.084   |
| Wide ResNet50 | 32        | INT8      | 5981.702 | 76.956   | 93.920   |
