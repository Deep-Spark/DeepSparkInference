# Res2Net50 (IGIE)

## Model Description

Res2Net50 is a convolutional neural network architecture that introduces the concept of "Residual-Residual Networks" (Res2Nets) to enhance feature representation and model expressiveness, particularly in image recognition tasks.The key innovation of Res2Net50 lies in its hierarchical feature aggregation mechanism, which enables the network to capture multi-scale features more effectively. Unlike traditional ResNet architectures, Res2Net50 incorporates multiple parallel pathways within each residual block, allowing the network to dynamically adjust the receptive field size and aggregate features across different scales.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.2.0     |  25.03  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.openmmlab.com/mmclassification/v0/res2net/res2net50-w14-s8_3rdparty_8xb32_in1k_20210927-bc967bf1.pth>

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
python3 ../../igie_common/export_mmcls.py --cfg mmpretrain/configs/res2net/res2net50-w14-s8_8xb32_in1k.py --weight res2net50-w14-s8_3rdparty_8xb32_in1k_20210927-bc967bf1.pth --output res2net50.onnx

# Use onnxsim optimize onnx model
onnxsim res2net50.onnx res2net50_opt.onnx

```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
export RUN_DIR=../../igie_common/
```

### FP16

```bash
# Accuracy
bash scripts/infer_res2net50_fp16_accuracy.sh
# Performance
bash scripts/infer_res2net50_fp16_performance.sh
```

## Model Results

| Model     | BatchSize | Precision | FPS      | Top-1(%) | Top-5(%) |
|-----------|-----------|-----------|----------|----------|----------|
| Res2Net50 | 32        | FP16      | 1641.961 | 78.139   | 93.826   |

## References

- [mmpretrain](https://github.com/open-mmlab/mmpretrain)
