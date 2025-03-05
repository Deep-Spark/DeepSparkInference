# MViTv2-base (IGIE)

## Model Description

MViTv2_base is an efficient multi-scale vision Transformer model designed specifically for image classification tasks. By employing a multi-scale structure and hierarchical representation, it effectively captures both global and local image features while maintaining computational efficiency. The MViTv2_base has demonstrated excellent performance on multiple standard datasets and is suitable for a variety of visual recognition tasks.

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.openmmlab.com/mmclassification/v0/mvit/mvitv2-base_3rdparty_in1k_20220722-9c4f0a17.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

pip3 install -r requirements.txt
```

### Model Conversion

```bash
# git clone mmpretrain
git clone -b v0.24.0 https://github.com/open-mmlab/mmpretrain.git

# export onnx model
python3 export.py --cfg mmpretrain/configs/mvit/mvitv2-base_8xb256_in1k.py --weight mvitv2-base_3rdparty_in1k_20220722-9c4f0a17.pth --output mvitv2_base.onnx

# Use onnxsim optimize onnx model
onnxsim mvitv2_base.onnx mvitv2_base_opt.onnx

```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
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
