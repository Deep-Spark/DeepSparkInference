# ConvNeXt Tiny (IGIE)

## Model Description

ConvNeXt is a modern convolutional neural network architecture proposed by Facebook AI Research, designed to optimize the performance of traditional CNNs by incorporating design principles from Transformers. ConvNeXt Tiny is the lightweight version of this series, specifically designed for resource-constrained devices.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.2.0     |  25.06  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/convnext_tiny-983f1562.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
pip3 install -r requirements.txt
```

### Model Conversion

```bash
python3 export.py --weight convnext_tiny-983f1562.pth --output convnext_tiny.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
```

### FP16

```bash
# Accuracy
bash scripts/infer_convnext_tiny_fp16_accuracy.sh
# Performance
bash scripts/infer_convnext_tiny_fp16_performance.sh
```

## Model Results

| Model          | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
| -------------- | --------- | --------- | ------- | -------- | -------- |
| ConvNeXt Tiny  | 32        | FP16      | 1128.96 | 82.104   | 95.919   |
