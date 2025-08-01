# ResNeSt50 (IGIE)

## Model Description

ResNeSt50 is a deep convolutional neural network model based on the ResNeSt architecture, specifically designed to enhance performance in visual recognition tasks such as image classification, object detection, instance segmentation, and semantic segmentation. ResNeSt stands for Split-Attention Networks, a modular network architecture that leverages channel-wise attention mechanisms across different network branches to capture cross-feature interactions and learn diverse representations.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.09 |
| MR-V100 | 4.2.0 | 25.03 |

## Model Preparation

### Prepare Resources

Pretrained model: <https://github.com/zhanghang1989/ResNeSt/releases/download/weights_step1/resnest50-528c19ca.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

pip3 install -r ../../igie_common/requirements.txt
pip3 install git+https://github.com/zhanghang1989/ResNeSt
```

### Model Conversion

```bash
# export onnx model
python3 ../../igie_common/export.py --model-name resnest50 --weight resnest50-528c19ca.pth --output resnest50.onnx

# Use onnxsim optimize onnx model
onnxsim resnest50.onnx resnest50_opt.onnx

```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
export RUN_DIR=../../igie_common/
```

### FP16

```bash
# Accuracy
bash scripts/infer_resnest50_fp16_accuracy.sh
# Performance
bash scripts/infer_resnest50_fp16_performance.sh
```

## Model Results

| Model     | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
| :----: | :----: | :----: | :----: | :----: | :----: |
| ResNeSt50 | 32        | FP16      | 344.453 | 80.93    | 95.347   |

## References

- [ResNeSt](https://github.com/zhanghang1989/ResNeSt)
