# FCOS (ixRT)

## Model Description

FCOS is an anchor-free model based on the Fully Convolutional Network (FCN) architecture for pixel-wise object detection. It implements a proposal-free solution and introduces the concept of centerness.
For more details, please refer to our [report on Arxiv](https://arxiv.org/abs/1904.01355).

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.09 |
| MR-V100 | 4.2.0 | 25.03 |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco-0a0d75a8.pth>

Dataset:

- <https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip> to download the labels dataset.
- <http://images.cocodataset.org/zips/val2017.zip> to download the validation dataset.
- <http://images.cocodataset.org/zips/train2017.zip> to download the train dataset.

```bash
unzip -q -d ./ coco2017labels.zip
unzip -q -d ./coco/images/ train2017.zip
unzip -q -d ./coco/images/ val2017.zip

coco
├── annotations
│   └── instances_val2017.json
├── images
│   ├── train2017
│   └── val2017
├── labels
│   ├── train2017
│   └── val2017
├── LICENSE
├── README.txt
├── test-dev2017.txt
├── train2017.cache
├── train2017.txt
├── val2017.cache
└── val2017.txt
```

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

MMDetection is an open source object detection toolbox based on PyTorch. It is a part of the OpenMMLab project.It is utilized for model conversion. In MMDetection, Execute model conversion command, and the checkpoints folder needs to be created, (mkdir checkpoints) in project

```bash
mkdir -p checkpoints
cd checkpoints
wget http://files.deepspark.org.cn:880/deepspark/fcos_opt.onnx
```

## Model Inference

```bash
export PROJ_DIR=./
export DATASETS_DIR=./coco/
export CHECKPOINTS_DIR=./checkpoints
export RUN_DIR=../../ixrt_common
```

### FP16

```bash
# Accuracy
bash scripts/infer_fcos_fp16_accuracy.sh
# Performance
bash scripts/infer_fcos_fp16_performance.sh
```

## Model Results

| Model | BatchSize | Precision | FPS   | MAP@0.5 | MAP@0.5:0.95 |
| ----- | --------- | --------- | ----- | ------- | ------------ |
| FCOS  | 1         | FP16      | 51.62 | 0.546   | 0.360        |
