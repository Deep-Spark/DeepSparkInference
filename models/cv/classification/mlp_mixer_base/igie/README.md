# MLP-Mixer Base (IGIE)

## Model Description

MLP-Mixer Base is a foundational model in the MLP-Mixer family, designed to use only MLP layers for vision tasks like image classification. Unlike CNNs and Vision Transformers, MLP-Mixer replaces both convolution and self-attention mechanisms with simple MLP layers to process spatial and channel-wise information independently.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.2.0     |  25.03  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.openmmlab.com/mmclassification/v0/mlp-mixer/mixer-base-p16_3rdparty_64xb64_in1k_20211124-1377e3e0.pth>

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
python3 ../../igie_common/export_mmcls.py --cfg mmpretrain/configs/mlp_mixer/mlp-mixer-base-p16_64xb64_in1k.py --weight mixer-base-p16_3rdparty_64xb64_in1k_20211124-1377e3e0.pth --output mlp_mixer_base.onnx

# Use onnxsim optimize onnx model
onnxsim mlp_mixer_base.onnx mlp_mixer_base_opt.onnx

```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
export RUN_DIR=../../igie_common/
```

### FP16

```bash
# Accuracy
bash scripts/infer_mlp_mixer_base_fp16_accuracy.sh
# Performance
bash scripts/infer_mlp_mixer_base_fp16_performance.sh
```

## Model Results

| Model           | BatchSize | Precision | FPS      | Top-1(%) | Top-5(%) |
| ----------------| --------- | --------- | -------- | -------- | -------- |
| MLP-Mixer-Base  | 32        | FP16      | 1477.15  | 72.545   | 90.035   |

## References

- [mmpretrain](https://github.com/open-mmlab/mmpretrain)
