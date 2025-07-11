# RegNet_x_1_6gf (IGIE)

## Model Description

RegNet is a family of models designed for image classification tasks, as described in the paper "Designing Network Design Spaces". The RegNet design space provides simple and fast networks that work well across a wide range of computational budgets.The architecture of RegNet models is based on the principle of designing network design spaces, which allows for a more systematic exploration of possible network architectures. This makes it easier to understand and modify the architecture.RegNet_x_1_6gf is a specific model within the RegNet family, designed for image classification tasks

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.09 |
| MR-V100 | 4.2.0 | 25.03 |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/regnet_x_1_6gf-a12f2b72.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
pip3 install -r ../../igie_common/requirements.txt
```

### Model Conversion

```bash
python3 ../../igie_common/export.py --model-name regnet_x_1_6gf --weight regnet_x_1_6gf-a12f2b72.pth --output regnet_x_1_6gf.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
export RUN_DIR=../../igie_common/
```

### FP16

```bash
# Accuracy
bash scripts/infer_regnet_x_1_6gf_fp16_accuracy.sh
# Performance
bash scripts/infer_regnet_x_1_6gf_fp16_performance.sh
```

## Model Results

| Model          | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
| :----: | :----: | :----: | :----: | :----: | :----: |
| RegNet_x_1_6gf | 32        | FP16      | 487.749 | 79.303   | 94.624   |
