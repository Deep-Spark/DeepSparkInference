# YOLOv4 (IxRT)

## Model Description

YOLOv4 employs a two-step process, involving regression for bounding box positioning and classification for object categorization. it amalgamates past YOLO family research contributions with novel features like WRC, CSP, CmBN, SAT, Mish activation, Mosaic data augmentation, DropBlock regularization, and CIoU loss.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.2.0     |  25.03  |

## Model Preparation

### Prepare Resources

Pretrained cfg: <https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg>
Pretrained model: <https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights>

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

### Model Conversion

```bash
# clone yolov4
git clone https://github.com/Tianxiaomo/pytorch-YOLOv4.git yolov4

# download weight
mkdir data
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights -P data

# export onnx model
python3 export.py --cfg yolov4/cfg/yolov4.cfg --weight data/yolov4.weights --batchsize 16 --output data/yolov4.onnx
mv yolov4_16_3_608_608_static.onnx data/yolov4.onnx

# Use onnxsim optimize onnx model
onnxsim data/yolov4.onnx data/yolov4_sim.onnx

# Make sure the dataset path is "data/coco"
```

## Model Inference

### FP16

```bash
# Accuracy
bash scripts/infer_yolov4_fp16_accuracy.sh
# Performance
bash scripts/infer_yolov4_fp16_performance.sh
```

### INT8

```bash
# Accuracy
bash scripts/infer_yolov4_int8_accuracy.sh
# Performance
bash scripts/infer_yolov4_int8_performance.sh
```

## Model Results

| Model  | BatchSize | Precision | FPS    | MAP@0.5 |
| ------ | --------- | --------- | ------ | ------- |
| YOLOv4 | 32        | FP16      | 303.27 | 0.730   |
| YOLOv4 | 32        | INT8      | 682.14 | 0.608   |

## References

- [darknet](https://github.com/AlexeyAB/darknet)
- [pytorch-YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4)
