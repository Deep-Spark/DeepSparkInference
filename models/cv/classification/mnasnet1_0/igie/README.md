# MNASNet1_0 (IGIE)

## Model Description

**MNASNet1_0** is a lightweight convolutional neural network designed using neural architecture search (NAS) to optimize both accuracy and latency for mobile devices. Its structure incorporates depthwise separable convolutions for efficiency, Squeeze-and-Excitation (SE) modules for enhanced feature representation, and compound scaling to balance width, depth, and resolution. This makes MNASNet1_0 highly efficient and ideal for resource-constrained and real-time applications.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.2.0     |  25.06  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/mnasnet1.0_top1_73.512-f206786ef8.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
pip3 install -r ../../igie_common/requirements.txt
```

### Model Conversion

```bash
python3 ../../igie_common/export.py --model-name mnasnet1_0 --weight mnasnet1.0_top1_73.512-f206786ef8.pth --output mnasnet1_0.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
export RUN_DIR=../../igie_common/
```

### FP16

```bash
# Accuracy
bash scripts/infer_mnasnet1_0_fp16_accuracy.sh
# Performance
bash scripts/infer_mnasnet1_0_fp16_performance.sh
```

## Model Results

| Model             | BatchSize | Precision | FPS      | Top-1(%) | Top-5(%) |
| ----------------- | --------- | --------- | -------- | -------- | -------- |
| MnasNet1_0        | 32        | FP16      | 5225.057 | 73.446   |  91.483  |
