# CSPDarkNet53 (IGIE)

## Model Description

CSPDarkNet53 is an enhanced convolutional neural network architecture that reduces redundant computations by integrating cross-stage partial network features and truncating gradient flow, thereby maintaining high accuracy while lowering computational costs.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.2.0     |  25.03  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.openmmlab.com/mmclassification/v0/cspnet/cspdarknet50_3rdparty_8xb32_in1k_20220329-bd275287.pth>

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
## cspdarknet50 is actually cspdarknet53
wget -O cspdarknet53_3rdparty_8xb32_in1k_20220329-bd275287.pth https://download.openmmlab.com/mmclassification/v0/cspnet/

python3 ../../igie_common/export_mmcls.py --cfg mmpretrain/configs/cspnet/cspdarknet50_8xb32_in1k.py --weight cspdarknet53_3rdparty_8xb32_in1k_20220329-bd275287.pth --output cspdarknet53.onnx

# Use onnxsim optimize onnx model
onnxsim cspdarknet53.onnx cspdarknet53_opt.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
export RUN_DIR=../../igie_common/
```

### FP16

```bash
# Accuracy
bash scripts/infer_cspdarknet53_fp16_accuracy.sh
# Performance
bash scripts/infer_cspdarknet53_fp16_performance.sh
```

## Model Results

| Model        | BatchSize | Precision | FPS      | Top-1(%) | Top-5(%) |
| ------------ | --------- | --------- | -------- | -------- | -------- |
| CSPDarkNet53 | 32        | FP16      | 3214.387 | 79.063   | 94.492   |

## References

- [mmpretrain](https://github.com/open-mmlab/mmpretrain)
