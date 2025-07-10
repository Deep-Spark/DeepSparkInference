# VAN_B0 (IGIE)

## Model Description

VAN-B0 is a lightweight visual attention network that combines convolution and attention mechanisms to enhance image classification performance. It achieves efficient feature capture by focusing on key areas and multi-scale feature extraction, making it suitable for running on resource-constrained devices.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.09 |
| MR-V100 | 4.2.0 | 25.06 |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.openmmlab.com/mmclassification/v0/van/van-tiny_8xb128_in1k_20220501-385941af.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

pip3 install -r ../../igie_common/requirements.txt
pip3 install mmcv==1.5.3 mmcls==0.24.0
```

### Model Conversion

```bash
# git clone mmpretrain
git clone -b v0.24.0 https://github.com/open-mmlab/mmpretrain.git

# export onnx model
python3 ../../igie_common/export_mmcls.py --cfg mmpretrain/configs/van/van-b0_8xb128_in1k.py --weight van-tiny_8xb128_in1k_20220501-385941af.pth --output van_b0.onnx

```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
export RUN_DIR=../../igie_common/
```

### FP16

```bash
# Accuracy
bash scripts/infer_van_b0_fp16_accuracy.sh
# Performance
bash scripts/infer_van_b0_fp16_performance.sh
```

## Model Results

| Model      | BatchSize | Precision | FPS      | Top-1(%) | Top-5(%) |
| ---------- | --------- | --------- | -------- | -------- | -------- |
| VAN_B0     | 32        | FP16      | 2155.35  | 72.079   | 91.209   |

## References

- [mmpretrain](https://github.com/open-mmlab/mmpretrain)
