# DBNet (ixRT)

## Model Description

DBNet (Differentiable Binarization Network) is a scene text detection model that uses a differentiable binarization process for robust text detection.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.4.0 | 26.06 |

## Model Preparation

### Prepare Resources

Pretrained models:
- r50_en_dbnet: <http://files.deepspark.org.cn:880/deepspark/data/checkpoints/r50_en_dbnet.onnx>

Dataset: ICDAR 2015 <http://files.deepspark.org.cn:880/deepspark/data/datasets/icdar_2015.zip>

### Install Dependencies

```bash
pip3 install shapely pyclipper opencv-python==4.6.0.66 tqdm
```

### Model Conversion

```bash
mkdir checkpoints
cd checkpoints
wget http://files.deepspark.org.cn:880/deepspark/data/checkpoints/r50_en_dbnet.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/path/to/icdar2015/
export CHECKPOINTS_DIR=./checkpoints
export RUN_DIR=./
```

### FP16

```bash
# Test ACC
bash scripts/infer_dbnet_fp16_accuracy.sh
# Test FPS
bash scripts/infer_dbnet_fp16_performance.sh
```

### INT8

```bash
# Test ACC
bash scripts/infer_dbnet_int8_accuracy.sh
# Test FPS
bash scripts/infer_dbnet_int8_performance.sh
```

## Model Results

| Model       | Backbone | BatchSize | Precision | FPS     | Hmean   |
| ----------- | -------- | --------- | --------- | ------- | ------- |
| DBNet       | r50_en   | 32        | FP16      | 143.85   | 0.803    |
| DBNet       | r50_en   | 32        | INT8      | 143.73   | 0.803    |