# Lightweight OpenPose

## Description

This work heavily optimizes the OpenPose approach to reach real-time inference on CPU with negliable accuracy drop. It detects a skeleton (which consists of keypoints and connections between them) to identify human poses for every person inside the image. The pose may contain up to 18 keypoints: ears, eyes, nose, neck, shoulders, elbows, wrists, hips, knees, and ankles. On COCO 2017 Keypoint Detection validation set this code achives 40% AP for the single scale inference (no flip or any post-processing done). 

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
pip3 install simplejson
pip3 install opencv-python==4.6.0.66
pip3 install mmcv==1.5.3
pip3 install pycocotools
```

### Download
- dataset: http://cocodataset.org/#download
- checkpoints: https://download.01.org/opencv/openvino_training_extensions/models/human_pose_estimation/checkpoint_iter_370000.pth

### Model Conversion

```bash
# export onnx model
git clone https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch.git
cd lightweight-human-pose-estimation.pytorch
mv scripts/convert_to_onnx.py .
python3 convert_to_onnx.py --checkpoint-path /Path/to/checkpoint_iter_370000.pth
cd ..
mkdir lightweight_openpose
onnxsim ./lightweight-human-pose-estimation.pytorch/human-pose-estimation.onnx ./lightweight_openpose/lightweight_openpose.onnx
```

## Inference

```bash
export DATASETS_DIR=/Path/to/coco_pose/
export CHECKPOINTS_DIR=/Path/to/lightweight_openpose/
```

### FP16

```bash
# Accuracy
bash scripts/infer_lightweight_openpose_fp16_accuracy.sh
# Performance
bash scripts/infer_lightweight_openpose_fp16_performance.sh
```

## Results

Model     |BatchSize  |Precision |FPS       |IOU@0.5   |IOU@0.5:0.95   |
----------|-----------|----------|----------|----------|---------------|
Lightweight OpenPose |    1    |   FP16   | 21030.833   |  0.660   |  0.401        |

## Reference

https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch