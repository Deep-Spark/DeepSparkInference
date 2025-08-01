# DenseNet201 (IGIE)

## Model Description

DenseNet201 is a deep convolutional neural network that stands out for its unique dense connection architecture, where each layer integrates features from all previous layers, effectively reusing features and reducing the number of parameters. This design not only enhances the network's information flow and parameter efficiency but also increases the model's regularization effect, helping to prevent overfitting. DenseNet201 consists of multiple dense blocks and transition layers, capable of capturing rich feature representations while maintaining computational efficiency, making it suitable for complex image recognition tasks.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.09 |
| MR-V100 | 4.2.0 | 25.03 |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/densenet201-c1103571.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
pip3 install -r ../../igie_common/requirements.txt
```

### Model Conversion

```bash
python3 ../../igie_common/export.py --model-name densenet201 --weight densenet201-c1103571.pth --output densenet201.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
export RUN_DIR=../../igie_common/
```

### FP16

```bash
# Accuracy
bash scripts/infer_densenet201_fp16_accuracy.sh
# Performance
bash scripts/infer_densenet201_fp16_performance.sh
```

## Model Results

| Model       | BatchSize | Precision | FPS      | Top-1(%) | Top-5(%) |
| ----------- | --------- | --------- | -------- | -------- | -------- |
| DenseNet201 | 32        | FP16      | 758.592  | 76.851   | 93.338   |
