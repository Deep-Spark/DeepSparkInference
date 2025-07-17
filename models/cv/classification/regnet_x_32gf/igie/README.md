# RegNet_x_32gf (IGIE)

## Model Description

RegNet_x_32gf is an efficient convolutional neural network model from Facebook AI's RegNet series, designed to provide flexible and efficient deep learning solutions through a regularized network architecture. The core idea of RegNet is to optimize network performance by designing the distribution rules for network width and depth, replacing traditional manual design. RegNet_x_32gf is a large variant with a computational complexity of 32 GFLOPs, making it suitable for high-performance image classification tasks.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0     |  25.09  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/regnet_x_32gf-9d47f8d0.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
pip3 install -r ../../igie_common/requirements.txt
```

### Model Conversion

```bash
python3 ../../igie_common/export.py --model-name regnet_x_32gf --weight regnet_x_32gf-9d47f8d0.pth --output regnet_x_32gf.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
export RUN_DIR=../../igie_common/
```

### FP16

```bash
# Accuracy
bash scripts/infer_regnet_x_32gf_fp16_accuracy.sh
# Performance
bash scripts/infer_regnet_x_32gf_fp16_performance.sh
```

## Model Results

| Model          | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
| :------------: | :-------: | :-------: | :-----: | :------: | :------: |
| RegNet_x_32gf  | 32        | FP16      | 449.752 | 80.594   | 95.216   |
