# DeiT-tiny (ixRT)

## Model Description

DeiT Tiny is a lightweight vision transformer designed for data-efficient learning. It achieves rapid training and high accuracy on small datasets through innovative attention distillation methods, while maintaining the simplicity and efficiency of the model.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.09 |
| MR-V100 | 4.2.0 | 25.03 |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.openmmlab.com/mmclassification/v0/deit/deit-tiny_pt-4xb256_in1k_20220218-13b382a0.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

pip3 install -r ../../ixrt_common/requirements.txt
pip3 install mmcv==1.5.3 mmcls==0.24.0
```

### Model Conversion

```bash
# git clone mmpretrain
git clone -b v0.24.0 https://github.com/open-mmlab/mmpretrain.git

mkdir checkpoints
# export onnx model
python3 ../../ixrt_common/export_mmcls.py --cfg mmpretrain/configs/deit/deit-tiny_pt-4xb256_in1k.py --weight deit-tiny_pt-4xb256_in1k_20220218-13b382a0.pth --output checkpoints/deit_tiny.onnx

# Use onnxsim optimize onnx model
onnxsim checkpoints/deit_tiny.onnx checkpoints/deit_tiny_opt.onnx
```

## Model Inference

```bash
export PROJ_DIR=./
export DATASETS_DIR=/path/to/imagenet_val/
export CHECKPOINTS_DIR=./checkpoints
export RUN_DIR=../../ixrt_common/
export CONFIG_DIR=../../ixrt_common/config/DEIT_TINY_CONFIG
```

### FP16

```bash

# Accuracy
bash scripts/infer_deit_tiny_fp16_accuracy.sh
# Performance
bash scripts/infer_deit_tiny_fp16_performance.sh

```

## Model Results

| Model     | BatchSize | Precision | FPS      | Top-1(%) | Top-5(%) |
| --------- | --------- | --------- | -------- | -------- | -------- |
| DeiT-tiny | 32        | FP16      | 1446.690 | 74.34    | 92.21    |

## References

- [mmpretrain](https://github.com/open-mmlab/mmpretrain)
