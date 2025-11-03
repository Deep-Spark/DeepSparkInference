# EfficientNet B7 (IGIE)

## Model Description

EfficientNet B7 is an advanced convolutional neural network model created by Google, which extends the Compound Scaling method to optimize the balance between network depth, width, and input resolution. It builds upon components such as Inverted Residual Blocks (MBConv), Squeeze-and-Excitation (SE) modules, and the Swish activation function. EfficientNet-B7 achieves state-of-the-art performance in areas like image classification and object detection. Although it demands substantial computational resources, its superior accuracy and efficiency render it well-suited for highly complex and demanding vision applications.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| MR-V100| 4.3.0     |  25.12  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/efficientnet_b7_lukemelas-c5b4e57e.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
pip3 install -r ../../igie_common/requirements.txt
```

### Model Conversion

```bash
python3 ../../igie_common/export.py --model-name efficientnet_b7 --weight efficientnet_b7_lukemelas-c5b4e57e.pth --output efficientnet_b7.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
export RUN_DIR=../../igie_common/
```

### FP16

```bash
# Accuracy
bash scripts/infer_efficientnet_b7_fp16_accuracy.sh
# Performance
bash scripts/infer_efficientnet_b7_fp16_performance.sh
```

## Model Results

| Model           | BatchSize | Precision | FPS      | Top-1(%) | Top-5(%) |
| --------------- | --------- | --------- | -------- | -------- | -------- |
| Efficientnet_b7 | 32        | FP16      | 532.606  | 73.902   | 91.531   |
