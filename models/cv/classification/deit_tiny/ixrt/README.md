# DeiT-tiny (IxRT)

## Model Description

DeiT Tiny is a lightweight vision transformer designed for data-efficient learning. It achieves rapid training and high accuracy on small datasets through innovative attention distillation methods, while maintaining the simplicity and efficiency of the model.

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

pip3 install -r requirements.txt
```

### Model Conversion

```bash
# git clone mmpretrain
git clone --depth 1 -b v1.1.0 https://github.com/open-mmlab/mmpretrain.git
(cd mmpretrain/ && python3 setup.py develop)

# export onnx model
python3 export.py --cfg mmpretrain/configs/deit/deit-tiny_4xb256_in1k.py --weight deit-tiny_pt-4xb256_in1k_20220218-13b382a0.pth --output deit_tiny.onnx

# Use onnxsim optimize onnx model
onnxsim deit_tiny.onnx deit_tiny_opt.onnx

```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
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
