# Twins_PCPVT (IGIE)

## Model Description

Twins_PCPVT Small is a lightweight vision transformer model that combines pyramid convolutions and self-attention mechanisms, designed for efficient image classification. It enhances the model's expressive capability through multi-scale feature extraction and convolutional embeddings.

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

Pretrained model: <https://download.openmmlab.com/mmclassification/v0/twins/twins-pcpvt-small_3rdparty_8xb128_in1k_20220126-ef23c132.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

pip3 install -r ../../igie_common/requirements.txt
pip3 install --no-build-isolation mmcv==1.5.3 mmcls==0.24.0
```

### Model Conversion

```bash
# git clone mmpretrain
git clone -b v0.24.0 https://github.com/open-mmlab/mmpretrain.git

# export onnx model
python3 ../../igie_common/export_mmcls.py --cfg mmpretrain/configs/twins/twins-pcpvt-small_8xb128_in1k.py --weight twins-pcpvt-small_3rdparty_8xb128_in1k_20220126-ef23c132.pth --output twins_pcpvt_small.onnx

# Use onnxsim optimize onnx model
onnxsim twins_pcpvt_small.onnx twins_pcpvt_small_opt.onnx

```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
export RUN_DIR=../../igie_common/
```

### FP16

```bash
# Accuracy
bash scripts/infer_twins_pcpvt_fp16_accuracy.sh
# Performance
bash scripts/infer_twins_pcpvt_fp16_performance.sh
```

## Model Results

| Model        | BatchSize | Precision | FPS      | Top-1(%) | Top-5(%) |
| ------------ | --------- | --------- | -------- | -------- | -------- |
| Twins_PCPVT  | 32        | FP16      | 1552.92  |  80.93   | 95.633   |

## References

- [mmpretrain](https://github.com/open-mmlab/mmpretrain)
