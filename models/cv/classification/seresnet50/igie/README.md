# SEResNet50 (IGIE)

## Model Description

SEResNet50 is an enhanced version of the ResNet50 network integrated with Squeeze-and-Excitation (SE) blocks, which strengthens the network's feature expression capability by explicitly emphasizing useful features and suppressing irrelevant ones. This improvement enables SEResNet50 to demonstrate higher accuracy in various visual recognition tasks compared to the standard ResNet50.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| MR-V100 | 4.2.0     |  25.03  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.openmmlab.com/mmclassification/v0/se-resnet/se-resnet50_batch256_imagenet_20200804-ae206104.pth>

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
python3 export.py --cfg mmpretrain/configs/seresnet/seresnet50_8xb32_in1k.py --weight se-resnet50_batch256_imagenet_20200804-ae206104.pth --output seresnet50.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
```

### FP16

```bash
# Accuracy
bash scripts/infer_seresnet_fp16_accuracy.sh
# Performance
bash scripts/infer_seresnet_fp16_performance.sh
```

## Model Results

| Model      | BatchSize | Precision | FPS      | Top-1(%) | Top-5(%) |
| ---------- | --------- | --------- | -------- | -------- | -------- |
| SEResNet50 | 32        | FP16      | 2548.268 | 77.709   | 93.812   |

## References

- [mmpretrain](https://github.com/open-mmlab/mmpretrain)
