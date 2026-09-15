# ResNet34 (IGIE)

## Model Description

ResNet-34 is a residual network with 34 layers. It uses shortcut connections to enable information flow across layers, mitigating the vanishing gradient problem and making deeper networks easier to train. It offers a good balance between classification accuracy and inference speed.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 5.0.0 | 26.09 |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/resnet34-b627a593.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
pip3 install -r ../../igie_common/requirements.txt
```

### Model Conversion

```bash
python3 ../../igie_common/export.py --model-name resnet34 --weight resnet34-b627a593.pth --output resnet34.onnx
```

## Model Inference

```bash
export PATH="/opt/sw_home/local/corex/bin:${PATH}"
export CUDA_PATH=/opt/sw_home/local/corex
export LD_LIBRARY_PATH="/usr/local/corex/lib64:/opt/sw_home/local/corex/lib64:/opt/sw_home/local/lib64:${LD_LIBRARY_PATH:-}"
export DATASETS_DIR=/Path/to/imagenet_val/
export RUN_DIR=../../igie_common/
```

### FP16

```bash
# Accuracy
bash scripts/infer_resnet34_fp16_accuracy.sh
# Performance
bash scripts/infer_resnet34_fp16_performance.sh
```

## Model Results

| Model    | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
| :----: | :----: | :----: | :----: | :----: | :----: |
| ResNet34 | 32        | FP16      | 6454.13 | 73.283   | 91.397   |
