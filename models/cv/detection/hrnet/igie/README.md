# HRNet

## Description

HRNet is an advanced deep learning architecture for human pose estimation, characterized by its maintenance of high-resolution representations throughout the entire network process, thereby avoiding the low-to-high resolution recovery step typical of traditional models. The network features parallel multi-resolution subnetworks and enriches feature representation through repeated multi-scale fusion, which enhances the accuracy of keypoint detection. Additionally, HRNet offers computational efficiency and has demonstrated superior performance over previous methods on several standard datasets.

## Setup

### Install

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

pip3 install -r requirements.txt
```

### Download

Pretrained model: <https://download.openmmlab.com/mmdetection/v2.0/hrnet/fcos_hrnetv2p_w18_gn-head_4x4_1x_coco/fcos_hrnetv2p_w18_gn-head_4x4_1x_coco_20201212_100710-4ad151de.pth>

Dataset: <http://images.cocodataset.org/zips/val2017.zip> to download the validation dataset.

### Model Conversion

```bash
# export onnx model
python3 export.py --weight fcos_hrnetv2p_w18_gn-head_4x4_1x_coco_20201212_100710-4ad151de.pth --cfg fcos_hrnetv2p-w18-gn-head_4xb4-1x_coco.py --output hrnet.onnx

# Use onnxsim optimize onnx model
onnxsim hrnet.onnx hrnet_opt.onnx
```

## Inference

```bash
export DATASETS_DIR=/Path/to/coco/
```

### FP16

```bash
# Accuracy
bash scripts/infer_hrnet_fp16_accuracy.sh
# Performance
bash scripts/infer_hrnet_fp16_performance.sh
```

## Results

Model  |BatchSize  |Precision |FPS       |IOU@0.5   |IOU@0.5:0.95   |
-------|-----------|----------|----------|----------|---------------|
HRNet  |    32     |   FP16   | 64.282   |  0.491   |  0.326        |

## Reference

mmdetection: <https://github.com/open-mmlab/mmdetection.git>
