# VGG19_BN (IGIE)

## Model Description

VGG19_BN is a variant of the VGG network, based on VGG19 with the addition of Batch Normalization (BN) layers. Batch Normalization is a technique used to accelerate training and improve model stability by normalizing the activation values of each layer. Compared to the original VGG19, VGG19_BN introduces Batch Normalization layers after each convolutional layer, further enhancing the model's training performance and generalization ability.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| MR-V100 | 4.2.0     |  25.06  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/vgg19_bn-c79401a0.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
pip3 install -r requirements.txt
```

### Model Conversion

```bash
python3 export.py --weight vgg19_bn-c79401a0.pth --output vgg19_bn.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
```

### FP16

```bash
# Accuracy
bash scripts/infer_vgg19_bn_fp16_accuracy.sh
# Performance
bash scripts/infer_vgg19_bn_fp16_performance.sh
```

## Model Results

|   Model  | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
|----------|-----------|-----------|---------|----------|----------|
| VGG19_BN | 32        | FP16      | 1654.42 | 74.216   | 91.809   |
