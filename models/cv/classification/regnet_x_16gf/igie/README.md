# RegNet_x_16gf (IGIE)

## Model Description

RegNet_x_16gf is a deep convolutional neural network from the RegNet family, introduced in the paper "Designing Network Design Spaces" by Facebook AI. RegNet models emphasize simplicity, efficiency, and scalability, and they systematically explore design spaces to achieve optimal performance.The x in RegNet_x_16gf indicates it belongs to the RegNetX series, which focuses on optimizing network width and depth, while 16gf refers to its computational complexity of approximately 16 GFLOPs. The model features linear width scaling, group convolutions, and bottleneck blocks, providing high accuracy while maintaining computational efficiency.


## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.2.0     |  25.03  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/regnet_x_16gf-2007eb11.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
pip3 install -r ../../igie_common/requirements.txt
```

### Model Conversion

```bash
python3 ../../igie_common/export.py --model-name regnet_x_16gf --weight regnet_x_16gf-2007eb11.pth --output regnet_x_16gf.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
export RUN_DIR=../../igie_common/
```

### FP16

```bash
# Accuracy
bash scripts/infer_regnet_x_16gf_fp16_accuracy.sh
# Performance
bash scripts/infer_regnet_x_16gf_fp16_performance.sh
```

## Model Results

| Model         | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
| :----: | :----: | :----: | :----: | :----: | :----: |
| RegNet_x_16gf | 32        | FP16      | 970.928 | 80.028   | 94.922   |
