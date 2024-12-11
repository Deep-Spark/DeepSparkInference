# RTMPose

## Description

RTMPose, a state-of-the-art framework developed by Shanghai AI Laboratory, excels in real-time multi-person pose estimation by integrating an innovative model architecture with the efficiency of the MMPose foundation. The framework's architecture is meticulously designed to enhance performance and reduce latency, making it suitable for a variety of applications where real-time analysis is crucial.

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
pip3 install mmdet==3.3.0
pip3 install mmpose==1.3.1
pip3 install mmdeploy==1.3.1
pip3 install mmengine==0.10.4
```

### Download

Pretrained model: <https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth>

Dataset: <http://images.cocodataset.org/zips/val2017.zip> to download the validation dataset.

## Model Conversion

```bash
# export onnx model

mkdir -p data/rtmpose

wget -P data/rtmpose/   https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth

python3 export.py --weight data/rtmpose/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth --cfg rtmpose-m_8xb256-420e_coco-256x192.py --input 1,3,256,192  --output data/rtmpose/rtmpose.onnx

# use onnxsim optimize onnx model
onnxsim data/rtmpose/rtmpose.onnx data/rtmpose/rtmpose_opt.onnx
```

## Inference

### FP16

```bash
python3 predict.py --model data/rtmpose/rtmpose_opt.onnx --precision fp16 --img_path demo/demo.jpg
```
