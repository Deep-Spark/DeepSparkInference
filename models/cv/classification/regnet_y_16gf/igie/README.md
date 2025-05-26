# RegNet_y_16gf (IGIE)

## Model Description

RegNet_y_16gf is an efficient convolutional neural network model in the RegNet family, proposed by Facebook AI, inspired by the paper *Designing Network Design Spaces*. The RegNet series systematically optimizes convolutional network structures through parameterized design methods, aiming to balance high performance and efficiency. RegNet_y_16gf belongs to the RegNet-Y branch, featuring approximately 16 GFLOPs of computational complexity, making it suitable for vision tasks in high-computation-resource scenarios.


## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.2.0     |  25.06  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/regnet_y_16gf-9e6ed7dd.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
pip3 install -r ../../igie_common/requirements.txt
```

### Model Conversion

```bash
python3 ../../igie_common/export.py --model-name regnet_y_16gf --weight regnet_y_16gf-9e6ed7dd.pth --output regnet_y_16gf.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
export RUN_DIR=../../igie_common/
```

### FP16

```bash
# Accuracy
bash scripts/infer_regnet_y_16gf_fp16_accuracy.sh
# Performance
bash scripts/infer_regnet_y_16gf_fp16_performance.sh
```

## Model Results

| Model         | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
| :----: | :----: | :----: | :----: | :----: | :----: |
| RegNet_y_16gf | 32        | FP16      | 689.918 | 80.398   | 95.214   |
