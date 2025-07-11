# ResNet18 (IGIE)

## Model Description

ResNet-18 is a relatively compact deep neural network.The ResNet-18 architecture consists of 18 layers, including convolutional, pooling, and fully connected layers. It incorporates residual blocks, a key innovation that utilizes shortcut connections to facilitate the flow of information through the network.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.09 |
| MR-V100 | 4.2.0 | 25.03 |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/resnet18-f37072fd.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
pip3 install -r ../../igie_common/requirements.txt
```

### Model Conversion

```bash
python3 ../../igie_common/export.py --model-name resnet18 --weight resnet18-f37072fd.pth --output resnet18.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
export RUN_DIR=../../igie_common/
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

| Model    | BatchSize | Precision | FPS      | Top-1(%) | Top-5(%) |
| :----: | :----: | :----: | :----: | :----: | :----: |
| ResNet18 | 32        | FP16      | 9592.98  | 69.77    | 89.09    |
| ResNet18 | 32        | INT8      | 21314.55 | 69.53    | 88.97    |
