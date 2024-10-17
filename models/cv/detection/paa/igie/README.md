# PAA

## Description

PAA (Probabilistic Anchor Assignment) is an algorithm for object detection that adaptively assigns positive and negative anchor samples using a probabilistic model. It employs a Gaussian mixture model to dynamically select positive and negative samples based on score distribution, avoiding the misassignment issues of traditional IoU threshold-based methods. PAA enhances detection accuracy, particularly in complex scenarios, and is compatible with existing detection frameworks.

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

Pretrained model: <https://download.openmmlab.com/mmdetection/v2.0/paa/paa_r50_fpn_1x_coco/paa_r50_fpn_1x_coco_20200821-936edec3.pth>

Dataset: <http://images.cocodataset.org/zips/val2017.zip> to download the validation dataset.

### Model Conversion

```bash
# export onnx model
python3 export.py --weight paa_r50_fpn_1x_coco_20200821-936edec3.pth --cfg paa_r50_fpn_1x_coco.py --output paa.onnx

# Use onnxsim optimize onnx model
onnxsim paa.onnx paa_opt.onnx
```

## Inference

```bash
export DATASETS_DIR=/Path/to/coco/
```

### FP16

```bash
# Accuracy
bash scripts/infer_paa_fp16_accuracy.sh
# Performance
bash scripts/infer_paa_fp16_performance.sh
```

## Results

Model  |BatchSize  |Precision |FPS       |IOU@0.5   |IOU@0.5:0.95   |
-------|-----------|----------|----------|----------|---------------|
PAA    |    32     |   FP16   | 138.414  |  0.555   |  0.381        |

## Reference

mmdetection: <https://github.com/open-mmlab/mmdetection.git>