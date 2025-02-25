# RetinaFace

## Description

RetinaFace is an efficient single-stage face detection model that employs a multi-task learning strategy to simultaneously predict facial locations, landmarks, and 3D facial shapes. It utilizes feature pyramids and context modules to extract multi-scale features and employs a self-supervised mesh decoder to enhance detection accuracy. RetinaFace demonstrates excellent performance on datasets like WIDER FACE, supports real-time processing, and its code and datasets are publicly available for researchers.

## Setup

### Install

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

pip3 install -r requirements.txt

python3 setup.py build_ext --inplace
```

### Download

Pretrained model: <https://github.com/biubug6/Face-Detector-1MB-with-landmark/raw/master/weights/mobilenet0.25_Final.pth>

Dataset: <http://shuoyang1213.me/WIDERFACE/> to download the validation dataset.

```bash
wget https://github.com/biubug6/Face-Detector-1MB-with-landmark/raw/master/weights/mobilenet0.25_Final.pth
```

### Model Conversion

```bash
# export onnx model
python3 torch2onnx.py --model mobilenet0.25_Final.pth --onnx_model mnetv1_retinaface.onnx

## Inference

```bash
export DATASETS_DIR=/Path/to/widerface/
export GT_DIR=../igie/ground_truth
```

### FP16

```bash
# Accuracy
bash scripts/infer_retinaface_fp16_accuracy.sh
# Performance
bash scripts/infer_retinaface_fp16_performance.sh
```

## Results

|   Model    | BatchSize | Precision |   FPS    | Easy AP(%) | Medium AP (%) | Hard AP(%) |
| :--------: | :-------: | :-------: | :------: | :--------: | :-----------: | :--------: |
| RetinaFace |    32     |   FP16    | 8536.367 |   80.84    |     69.34     |   37.31    |

## Reference

Face-Detector-1MB-with-landmark: <https://github.com/biubug6/Face-Detector-1MB-with-landmark>
