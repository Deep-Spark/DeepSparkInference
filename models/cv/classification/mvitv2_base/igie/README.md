# MViTv2-base (IGIE)

## Model Description

MViTv2_base is an efficient multi-scale vision Transformer model designed specifically for image classification tasks. By employing a multi-scale structure and hierarchical representation, it effectively captures both global and local image features while maintaining computational efficiency. The MViTv2_base has demonstrated excellent performance on multiple standard datasets and is suitable for a variety of visual recognition tasks.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | dev-only     |  25.12  |
| MR-V100 | 4.2.0     |  25.03  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.openmmlab.com/mmclassification/v0/mvit/mvitv2-base_3rdparty_in1k_20220722-9c4f0a17.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
pip3 install -r ../../igie_common/requirements.txt
pip3 install mmcv==1.5.3 mmcls==0.24.0
```

### Model Conversion

```bash
# git clone mmpretrain
git clone -b v0.24.0 https://github.com/open-mmlab/mmpretrain.git

# export onnx model
python3 ../../igie_common/export_mmcls.py --cfg mmpretrain/configs/mvit/mvitv2-base_8xb256_in1k.py --weight mvitv2-base_3rdparty_in1k_20220722-9c4f0a17.pth --output mvitv2_base.onnx

# Use onnxsim optimize onnx model
onnxsim mvitv2_base.onnx mvitv2_base_opt.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet-val/
export RUN_DIR=../../igie_common/
```

### FP16

```bash
# Accuracy
bash scripts/infer_mvitv2_base_fp16_accuracy.sh
# Performance
bash scripts/infer_mvitv2_base_fp16_performance.sh
```

## Model Results

| Model       | BatchSize | Precision | FPS      | Top-1(%) | Top-5(%) |
| ----------- | --------- | --------- | -------- | -------- | -------- |
| MViTv2-base | 16        | FP16      | 58.76    | 84.226   | 96.848   |

## References

- [mmpretrain](https://github.com/open-mmlab/mmpretrain)
