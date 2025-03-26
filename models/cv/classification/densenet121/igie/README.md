# DenseNet121 (IGIE)

## Model Description

DenseNet-121 is a convolutional neural network architecture that belongs to the family of Dense Convolutional Networks.The network consists of four dense blocks, each containing a varying number of densely connected convolutional layers. Transition layers with pooling operations reduce the spatial dimensions between dense blocks.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| MR-V100 | 4.2.0     |  25.03  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/densenet121-a639ec97.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
pip3 install -r requirements.txt
```

### Model Conversion

```bash
python3 export.py --weight densenet121-a639ec97.pth --output densenet121.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
```

### FP16

```bash
# Accuracy
bash scripts/infer_densenet121_fp16_accuracy.sh
# Performance
bash scripts/infer_densenet121_fp16_performance.sh
```

## Model Results

| Model       | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
|-------------|-----------|-----------|---------|----------|----------|
| DenseNet121 | 32        | FP16      | 2199.75 | 74.40    | 91.931   |
