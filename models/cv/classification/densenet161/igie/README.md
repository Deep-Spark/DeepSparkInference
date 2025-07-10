# DenseNet161 (IGIE)

## Model Description

DenseNet161 is a convolutional neural network architecture that belongs to the family of Dense Convolutional Networks (DenseNets). Introduced as an extension to the previous DenseNet models, DenseNet161 offers improved performance and deeper network capacity, making it suitable for various computer vision tasks.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.09 |
| MR-V100 | 4.2.0 | 25.03 |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/densenet161-8d451a50.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
pip3 install -r ../../igie_common/requirements.txt
```

### Model Conversion

```bash
python3 ../../igie_common/export.py --model-name densenet161 --weight densenet161-8d451a50.pth --output densenet161.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
export RUN_DIR=../../igie_common/
```

### FP16

```bash
# Accuracy
bash scripts/infer_densenet161_fp16_accuracy.sh
# Performance
bash scripts/infer_densenet161_fp16_performance.sh
```

## Model Results

| Model       | BatchSize | Precision | FPS    | Top-1(%) | Top-5(%) |
| ----------- | --------- | --------- | ------ | -------- | -------- |
| DenseNet161 | 32        | FP16      | 520.26 | 77.095   | 93.526   |
