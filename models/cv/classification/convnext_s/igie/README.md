# ConvNext-S (IGIE)

## Model Description

ConvNeXt-S is a small-sized model in the ConvNeXt family, designed to balance performance and computational complexity. With 50.22M parameters and 8.69G FLOPs, it achieves 83.13% Top-1 accuracy on ImageNet-1k. Modernized from traditional ConvNets, ConvNeXt-S incorporates features such as large convolutional kernels (7x7), LayerNorm, and GELU activations, making it highly efficient and scalable.

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.openmmlab.com/mmclassification/v0/convnext/convnext-small_3rdparty_32xb128_in1k_20220124-d39b5192.pth>

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
python3 export.py --cfg mmpretrain/configs/convnext/convnext-small_32xb128_in1k.py --weight convnext-small_3rdparty_32xb128_in1k_20220124-d39b5192.pth --output convnext_s.onnx

# Use onnxsim optimize onnx model
onnxsim convnext_s.onnx convnext_s_opt.onnx

```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
```

### FP16

```bash
# Accuracy
bash scripts/infer_convnext_s_fp16_accuracy.sh
# Performance
bash scripts/infer_convnext_s_fp16_performance.sh
```

## Model Results

| Model        | BatchSize | Precision | FPS      | Top-1(%) | Top-5(%) |
| ------------ | --------- | --------- | -------- | -------- | -------- |
| ConvNext-S   | 32        | FP16      | 728.32   | 82.786   | 96.415   |

## References

- [ConvNext-S](https://github.com/open-mmlab/mmpretrain)
