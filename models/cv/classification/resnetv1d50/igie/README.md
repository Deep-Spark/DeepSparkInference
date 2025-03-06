# ResNetV1D50 (IGIE)

## Model Description

ResNetV1D50 is an enhanced version of ResNetV1-50 that incorporates changes like dilated convolutions and adjusted downsampling, leading to better performance in large-scale image classification tasks. Its ability to capture richer image features makes it a popular choice in deep learning models.

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.openmmlab.com/mmclassification/v0/resnet/resnetv1d50_b32x8_imagenet_20210531-db14775a.pth>

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
python3 export.py --cfg mmpretrain/configs/resnet/resnetv1d50_b32x8_imagenet.py --weight resnetv1d50_b32x8_imagenet_20210531-db14775a.pth --output resnetv1d50.onnx

```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
```

### FP16

```bash
# Accuracy
bash scripts/infer_resnetv1d50_fp16_accuracy.sh
# Performance
bash scripts/infer_resnetv1d50_fp16_performance.sh
```

## Model Results

| Model       | BatchSize | Precision | FPS      | Top-1(%) | Top-5(%) |
| ----------- | --------- | --------- | -------- | -------- | -------- |
| ResNetV1D50 | 32        | FP16      | 4017.92  | 77.517   | 93.538   |

## References

- [mmpretrain](https://github.com/open-mmlab/mmpretrain)
