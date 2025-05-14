# SqueezeNet1_1 (IGIE)

## Model Description

SqueezeNet 1.1 is an improved version of SqueezeNet, designed for efficient computation in resource-constrained environments. It retains the core idea of SqueezeNet ,  significantly reduce the number of parameters and model size while maintaining high classification performance. Compared to SqueezeNet 1.0, version 1.1 further optimizes the network structure by reducing the number of channels, adjusting the strides of certain convolution layers, and simplifying the model design, resulting in faster inference and higher efficiency. 

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.2.0     |  25.03  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/squeezenet1_1-b8a52dc0.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
pip3 install -r requirements.txt
```

### Model Conversion

```bash
python3 export.py --weight squeezenet1_1-b8a52dc0.pth --output squeezenet1_1.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
```

### FP16

```bash
# Accuracy
bash scripts/infer_squeezenet_v1_1_fp16_accuracy.sh
# Performance
bash scripts/infer_squeezenet_v1_1_fp16_performance.sh
```

## Model Results

| Model           | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
|-----------------|-----------|-----------|---------|----------|----------|
| Squeezenet_v1_1 | 32        | FP16      | 14815.8 | 58.14    | 80.58    |
