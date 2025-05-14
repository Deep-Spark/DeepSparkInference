# ShuffleNetV2_x1_0 (IGIE)

## Model Description

ShuffleNet V2_x1_0 is an efficient convolutional neural network (CNN) architecture that emphasizes a balance between computational efficiency and accuracy, particularly suited for deployment on mobile and embedded devices. The model refines the ShuffleNet series by introducing structural innovations that enhance feature reuse and reduce redundancy, all while maintaining simplicity and performance.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.2.0     |  25.03  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
pip3 install -r ../../igie_common/requirements.txt
```

### Model Conversion

```bash
python3 ../../igie_common/export.py --model-name shufflenet_v2_x1_0 --weight shufflenetv2_x1-5666bf0f80.pth --output shufflenetv2_x1_0.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
export RUN_DIR=../../igie_common/
```

### FP16

```bash
# Accuracy
bash scripts/infer_shufflenetv2_x1_0_fp16_accuracy.sh
# Performance
bash scripts/infer_shufflenetv2_x1_0_fp16_performance.sh
```

## Model Results

| Model             | BatchSize | Precision | FPS      | Top-1(%) | Top-5(%) |
| ----------------- | --------- | --------- | -------- | -------- | -------- |
| ShuffleNetV2_x1_0 | 32        | FP16      | 8232.980 | 69.308   | 88.302   |
