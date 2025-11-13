# RTMPose (ixRT)

## Model Description

RTMPose, a state-of-the-art framework developed by Shanghai AI Laboratory, excels in real-time multi-person pose estimation by integrating an innovative model architecture with the efficiency of the MMPose foundation. The framework's architecture is meticulously designed to enhance performance and reduce latency, making it suitable for a variety of applications where real-time analysis is crucial.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.09 |
| MR-V100 | 4.2.0 | 25.03 |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth>

Dataset:
  - <https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip> to download the labels dataset.
  - <http://images.cocodataset.org/zips/val2017.zip> to download the validation dataset.
  - <http://images.cocodataset.org/zips/train2017.zip> to download the train dataset.
  - <http://images.cocodataset.org/annotations/annotations_trainval2017.zip> to download the annotations dataset.

```bash
unzip -q -d ./ coco2017labels.zip
unzip -q -d ./coco/images/ train2017.zip
unzip -q -d ./coco/images/ val2017.zip
unzip -q -d ./coco annotations_trainval2017.zip

coco
├── annotations
│   └── instances_train2017.json
│   └── instances_val2017.json
│   └── captions_train2017.json
│   └── captions_val2017.json
│   └── person_keypoints_train2017.json
│   └── person_keypoints_val2017.json
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
# before install mmpose==1.3.1 need to install chchumpy==0.70 which is too older that is not compatible with newer Python versions or pip
# so need to downgrade pip to version 20.2.4
pip install pip==20.2.4
pip install mmpose==1.3.1
pip install --upgrade pip
pip install -r requirements.txt
```

## Model Conversion

```bash
# export onnx model

mkdir -p data/rtmpose

wget -P data/rtmpose/ https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth

python3 export.py --weight data/rtmpose/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth --cfg rtmpose-m_8xb256-420e_coco-256x192.py --input 1,3,256,192  --output data/rtmpose/rtmpose.onnx

# use onnxsim optimize onnx model
onnxsim data/rtmpose/rtmpose.onnx data/rtmpose/rtmpose_opt.onnx
```

## Model Inference

### FP16

```bash
python3 predict.py --model data/rtmpose/rtmpose_opt.onnx --precision fp16 --img_path demo/demo.jpg
```
