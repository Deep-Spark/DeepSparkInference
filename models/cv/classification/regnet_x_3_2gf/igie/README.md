# RegNet_x_3_2gf (IGIE)

## Model Description

RegNet_x_3_2gf is a model from the RegNet series, inspired by the paper *Designing Network Design Spaces*. It leverages a parameterized network design approach to optimize convolutional neural network structures. With features like linear width scaling, group convolutions, and bottleneck blocks, it is particularly suited for efficient applications in scenarios with moderate computational resources.

## Supported Environments

| GPU | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release | Branch |
| :----: | :----: | :----: | :----: |
| MR-V100 | 4.4.0 | 26.03 | release/26.03 |
| MR-V100 | 4.3.0 | 25.12 | release/25.12 |

> **Note:** 请切换到与您的 SDK 版本对应的 Release 分支进行测试。请勿直接在 master 分支上运行测试，因为 master 分支可能包含与您的本地 SDK 版本不兼容的最新更改。
>
> 切换分支命令示例：`git checkout release/26.03`

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/regnet_x_3_2gf-f342aeae.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
pip3 install -r ../../igie_common/requirements.txt
```

### Model Conversion

```bash
python3 ../../igie_common/export.py --model-name regnet_x_3_2gf --weight regnet_x_3_2gf-f342aeae.pth --output regnet_x_3_2gf.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
export RUN_DIR=../../igie_common/
```

### FP16

```bash
# Accuracy
bash scripts/infer_regnet_x_3_2gf_fp16_accuracy.sh
# Performance
bash scripts/infer_regnet_x_3_2gf_fp16_performance.sh
```

## Model Results

| Model          | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
| :----: | :----: | :----: | :----: | :----: | :----: |
| RegNet_x_3_2gf | 32        | FP16      | 2096.88 | 78.333   | 93.962   |
