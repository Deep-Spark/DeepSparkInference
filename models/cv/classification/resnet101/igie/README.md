# ResNet101 (IGIE)

## Model Description

ResNet101 is a convolutional neural network architecture that belongs to the ResNet (Residual Network) family.With a total of 101 layers, ResNet101 comprises multiple residual blocks, each containing convolutional layers with batch normalization and rectified linear unit (ReLU) activations. These residual blocks allow the network to effectively capture complex features at different levels of abstraction, leading to superior performance on image recognition tasks.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.2.0     |  25.03  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/resnet101-63fe2227.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
pip3 install -r requirements.txt
```

### Model Conversion

```bash
python3 export.py --weight resnet101-63fe2227.pth --output resnet101.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
```

### FP16

```bash
# Accuracy
bash scripts/infer_resnet101_fp16_accuracy.sh
# Performance
bash scripts/infer_resnet101_fp16_performance.sh
```

### INT8

```bash
# Accuracy
bash scripts/infer_resnet101_int8_accuracy.sh
# Performance
bash scripts/infer_resnet101_int8_performance.sh
```

## Model Results

| Model     | BatchSize | Precision | FPS      | Top-1(%) | Top-5(%) |
|-----------|-----------|-----------|----------|----------|----------|
| ResNet101 | 32        | FP16      | 2507.074 | 77.331   | 93.520   |
| ResNet101 | 32        | INT8      | 5458.890 | 76.719   | 93.348   |
