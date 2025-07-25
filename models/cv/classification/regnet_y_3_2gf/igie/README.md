# RegNet_y_3_2gf (IGIE)

## Model Description

RegNet_y_3_2gf is a lightweight deep learning model designed with a regularized architecture, integrating Bottleneck Blocks and SE modules to enhance feature representation. With lower computational complexity, it is well-suited for mid-performance image classification tasks in resource-constrained environments.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0     |  25.09  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/regnet_y_3_2gf-b5a9779c.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
pip3 install -r ../../igie_common/requirements.txt
```

### Model Conversion

```bash
python3 ../../igie_common/export.py --model-name regnet_y_3_2gf --weight regnet_y_3_2gf-b5a9779c.pth --output regnet_y_3_2gf.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
export RUN_DIR=../../igie_common/
```

### FP16

```bash
# Accuracy
bash scripts/infer_regnet_y_3_2gf_fp16_accuracy.sh
# Performance
bash scripts/infer_regnet_y_3_2gf_fp16_performance.sh
```

## Model Results

| Model          | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
| :----: | :----: | :----: | :----: | :----: | :----: |
| RegNet_y_3_2gf | 32        | FP16      | 1548.577| 78.913   | 94.542   |
