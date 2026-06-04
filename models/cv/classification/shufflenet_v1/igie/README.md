# ShuffleNetV1 (IGIE)

## Model Description

ShuffleNet V1 is a lightweight neural network architecture primarily used for image classification and object detection tasks.
It uses techniques such as deep separable convolution and channel shuffle to reduce the number of parameters and computational complexity of the model, thereby achieving low computational resource consumption while maintaining high accuracy.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.4.0 | 26.06 |


## Model Preparation

### Prepare Resources

Pretrained model: <https://download.openmmlab.com/mmclassification/v0/shufflenet_v1/shufflenet_v1_batch1024_imagenet_20200804-5d6cec73.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
pip3 install -r ../../igie_common/requirements.txt
pip3 install --no-build-isolation mmcv==1.5.3 mmcls==0.24.0
```

### Model Conversion

```bash
git clone -b v0.24.0 https://github.com/open-mmlab/mmpretrain.git

python3 ../../igie_common/export_mmcls.py   \
    --cfg ./mmpretrain/configs/shufflenet_v1/shufflenet-v1-1x_16xb64_in1k.py  \
    --weight  ./shufflenet_v1_batch1024_imagenet_20200804-5d6cec73.pth \
    --output shufflenetv1.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
export RUN_DIR=../../igie_common/
```

### FP16

```bash
# Accuracy
bash scripts/infer_shufflenet_v1_fp16_accuracy.sh
# Performance
bash scripts/infer_shufflenet_v1_fp16_performance.sh
```

## Model Results

| Model        | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
| :----: | :----: | :----: | :----: | :----: | :----: |
| ShuffleNetV1 | 32        | FP16      | 8867.570 | 68.098    | 87.802    |
