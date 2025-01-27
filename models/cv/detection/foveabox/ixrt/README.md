# FoveaBox

## Description

FoveaBox is an advanced anchor-free object detection framework that enhances accuracy and flexibility by directly predicting the existence and bounding box coordinates of objects. Utilizing a Feature Pyramid Network (FPN), it adeptly handles targets of varying scales, particularly excelling with objects of arbitrary aspect ratios. FoveaBox also demonstrates robustness against image deformations.

## Setup

### Install

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-dev

pip3 install tqdm
pip3 install onnx
pip3 install onnxsim
pip3 install ultralytics
pip3 install pycocotools
pip3 install mmdeploy
pip3 install mmdet
pip3 install opencv-python==4.6.0.66
```

### Download

Pretrained model: <https://download.openmmlab.com/mmdetection/v2.0/foveabox/fovea_r50_fpn_4x4_1x_coco/fovea_r50_fpn_4x4_1x_coco_20200219-ee4d5303.pth>

Dataset: <http://images.cocodataset.org/zips/val2017.zip> to download the validation dataset.

### Model Conversion

```bash
# export onnx model
python3 export.py --weight fovea_r50_fpn_4x4_1x_coco_20200219-ee4d5303.pth --cfg fovea_r50_fpn_4xb4-1x_coco.py --output foveabox.onnx

# Use onnxsim optimize onnx model
onnxsim foveabox.onnx foveabox_opt.onnx
```

## Inference

```bash
export DATASETS_DIR=/Path/to/coco/
```

### FP16

```bash
# Accuracy
bash scripts/infer_foveabox_fp16_accuracy.sh
# Performance
bash scripts/infer_foveabox_fp16_performance.sh
```

## Results

Model    |BatchSize  |Precision |FPS       |IOU@0.5   |IOU@0.5:0.95   |
---------|-----------|----------|----------|----------|---------------|
FoveaBox |    32     |   FP16   | 181.304  |  0.531   |  0.346        |

## Reference

mmdetection: <https://github.com/open-mmlab/mmdetection.git>
