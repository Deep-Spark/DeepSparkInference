# EfficientNet B6 (IGIE)

## Model Description

EfficientNet B6 is a high-performance convolutional neural network model developed by Google, utilizing Compound Scaling to balance the scaling of depth, width, and resolution. It incorporates Inverted Residual Blocks, Squeeze-and-Excitation (SE) modules, and the Swish activation function. EfficientNet-B6 excels in tasks like image classification and object detection. While it requires significant computational resources, its precision and efficiency make it an ideal choice for complex vision tasks.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| MR-V100| 4.2.0     |  25.06  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/efficientnet_b6_lukemelas-24a108a5.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
pip3 install -r requirements.txt
```

### Model Conversion

```bash
python3 export.py --weight efficientnet_b6_lukemelas-24a108a5.pth --output efficientnet_b6.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
```

### FP16

```bash
# Accuracy
bash scripts/infer_efficientnet_b6_fp16_accuracy.sh
# Performance
bash scripts/infer_efficientnet_b6_fp16_performance.sh
```

## Model Results

| Model           | BatchSize | Precision | FPS      | Top-1(%) | Top-5(%) |
| --------------- | --------- | --------- | -------- | -------- | -------- |
| Efficientnet_b6 | 32        | FP16      | 523.225  | 74.388   | 91.835   |
