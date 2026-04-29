# DDRNet (ixRT)

## Model Description

DDRNet (Dual Resolution Network) is a real-time semantic segmentation network that learns rich representations through bilateral detail preservation and deep aggregation for high-resolution image understanding.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.4.0 | 26.06 |

## Model Preparation

### Prepare Resources

Pretrained model: <http://files.deepspark.org.cn:880/deepspark/data/checkpoints/ddrnet23.onnx>

Dataset: <https://www.cityscapes-dataset.com> to download the dataset.

### Install Dependencies

```bash
pip3 install xtcocotools tqdm munkres onnxsim opencv-python==4.6.0.66
```

### Model Conversion

```bash
mkdir checkpoints
cd checkpoints
wget http://files.deepspark.org.cn:880/deepspark/data/checkpoints/ddrnet23.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/cityscapes/
export CHECKPOINTS_DIR=./checkpoints
export RUN_DIR=./
```

### FP16

```bash
# Test ACC (mIoU)
bash scripts/infer_ddrnet_fp16_accuracy.sh
# Test FPS
bash scripts/infer_ddrnet_fp16_performance.sh
```

### INT8

```bash
# Test ACC (mIoU)
bash scripts/infer_ddrnet_int8_accuracy.sh
# Test FPS
bash scripts/infer_ddrnet_int8_performance.sh
```

## Model Results

| Model  | BatchSize | Precision | FPS     | mIoU(%) | mAcc(%) |
| ------ | --------- | --------- | ------- | ------- | ------- |
| DDRNet | 4         | FP16      |  98.278  | 12.8    | 25.8    |
| DDRNet | 4         | INT8      | 123.94  | 12.9    | 25.6    |