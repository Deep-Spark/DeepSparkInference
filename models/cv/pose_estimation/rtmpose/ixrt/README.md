# RTMPose (IxRT)

## Model Description

RTMPose, a state-of-the-art framework developed by Shanghai AI Laboratory, excels in real-time multi-person pose estimation by integrating an innovative model architecture with the efficiency of the MMPose foundation. The framework's architecture is meticulously designed to enhance performance and reduce latency, making it suitable for a variety of applications where real-time analysis is crucial.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.2.0     |  25.03  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth>

Dataset: <http://images.cocodataset.org/zips/val2017.zip> to download the validation dataset.

### Install Dependencies

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

pip3 install -r requirements.txt
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
