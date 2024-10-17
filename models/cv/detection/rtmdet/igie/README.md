# RTMDet

## Description

RTMDet, presented by the Shanghai AI Laboratory, is a novel framework for real-time object detection that surpasses the efficiency of the YOLO series. The model's architecture is meticulously crafted for optimal efficiency, employing a basic building block consisting of large-kernel depth-wise convolutions in both the backbone and neck, which enhances the model's ability to capture global context.

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
pip3 install mmdeploy==1.3.1
pip3 install mmengine==0.10.4
```

### Download
<<<<<<< HEAD

=======
>>>>>>> e135208fbca20b05e53a48d9706c146f9a07ef1c
Pretrained model: <https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth>

Dataset: <http://images.cocodataset.org/zips/val2017.zip> to download the validation dataset.

```bash
wget https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth
```

### Model Conversion

```bash
# export onnx model
python3 export.py --weight rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth --cfg rtmdet_nano_320-8xb32_coco-person.py --output rtmdet.onnx

# use onnxsim optimize onnx model
onnxsim rtmdet.onnx rtmdet_opt.onnx
```

## Inference

```bash
export DATASETS_DIR=/Path/to/coco/
```

### FP16

```bash
# Accuracy
bash scripts/infer_rtmdet_fp16_accuracy.sh
# Performance
bash scripts/infer_rtmdet_fp16_performance.sh
```

## Results
Model     |BatchSize  |Precision |FPS       |IOU@0.5   |IOU@0.5:0.95   |
----------|-----------|----------|----------|----------|---------------|
RTMDet    |    32     |   FP16   | 2627.15  |  0.619   |  0.403        |

## Reference

mmdetection: <https://github.com/open-mmlab/mmdetection.git>
