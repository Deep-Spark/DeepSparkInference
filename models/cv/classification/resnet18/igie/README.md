# ResNet18

## Model Description

ResNet-18 is a relatively compact deep neural network.The ResNet-18 architecture consists of 18 layers, including convolutional, pooling, and fully connected layers. It incorporates residual blocks, a key innovation that utilizes shortcut connections to facilitate the flow of information through the network.

## Model Preparation

### Install Dependencies

```bash
pip3 install -r requirements.txt
```

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/resnet18-f37072fd.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Model Conversion

```bash
python3 export.py --weight resnet18-f37072fd.pth --output resnet18.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
```

### FP16

```bash
# Accuracy
bash scripts/infer_resnet18_fp16_accuracy.sh
# Performance
bash scripts/infer_resnet18_fp16_performance.sh
```

### INT8

```bash
# Accuracy
bash scripts/infer_resnet18_int8_accuracy.sh
# Performance
bash scripts/infer_resnet18_int8_performance.sh
```

## Model Results

Model    |BatchSize  |Precision |FPS       |Top-1(%)  |Top-5(%)
---------|-----------|----------|----------|----------|--------
ResNet18 |    32     |   FP16   | 9592.98  |  69.77   | 89.09
ResNet18 |    32     |   INT8   | 21314.55 |  69.53   | 88.97
