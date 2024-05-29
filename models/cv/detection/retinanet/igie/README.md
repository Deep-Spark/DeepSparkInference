# RetinaNet

## Description

RetinaNet, an innovative object detector, challenges the conventional trade-off between speed and accuracy in the realm of computer vision. Traditionally, two-stage detectors, exemplified by R-CNN, achieve high accuracy by applying a classifier to a limited set of candidate object locations. In contrast, one-stage detectors, like RetinaNet, operate over a dense sampling of possible object locations, aiming for simplicity and speed.

## Setup

### Install

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-dev

pip3 install onnx
pip3 install tqdm
pip3 install onnxsim
pip3 install mmdet
pip3 install mmdeploy
pip3 install mmengine
```

### Download

Pretrained model: <https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth>

Dataset: <http://images.cocodataset.org/zips/val2017.zip> to download the validation dataset.

### Model Conversion

```bash
# export onnx model
python3 export.py --weight retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth --cfg retinanet_r50_fpn_1x_coco.py --output retinanet.onnx

# Use onnxsim optimize onnx model
onnxsim retinanet.onnx retinanet_opt.onnx
```

## Inference

```bash
export DATASETS_DIR=/Path/to/coco/
```

### FP16

```bash
# Accuracy
bash scripts/infer_retinanet_fp16_accuracy.sh
# Performance
bash scripts/infer_retinanet_fp16_performance.sh
```

## Results

Model     |BatchSize  |Precision |FPS       |IOU@0.5   |IOU@0.5:0.95   |
----------|-----------|----------|----------|----------|---------------|
RetinaNet |    32     |   FP16   | 160.52   |  0.515   |  0.335        |

## Reference

mmdetection: <https://github.com/open-mmlab/mmdetection.git>
