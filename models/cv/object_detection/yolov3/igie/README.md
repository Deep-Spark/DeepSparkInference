# YOLOv3 (IGIE)

## Model Description

YOLOv3 is a influential object detection algorithm.The key innovation of YOLOv3 lies in its ability to efficiently detect and classify objects in real-time with a single pass through the neural network. YOLOv3 divides an input image into a grid and predicts bounding boxes, class probabilities, and objectness scores for each grid cell.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.09 |
| MR-V100 | 4.2.0 | 25.03 |

## Model Preparation

### Prepare Resources

Pretrained model: <https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov3.pt>

Dataset:

- <https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip> to download the labels dataset.
- <http://images.cocodataset.org/zips/val2017.zip> to download the validation dataset.
- <http://images.cocodataset.org/zips/train2017.zip> to download the train dataset.

```bash
unzip -q -d ./ coco2017labels.zip
unzip -q -d ./coco/images/ train2017.zip
unzip -q -d ./coco/images/ val2017.zip

coco
├── annotations
│   └── instances_val2017.json
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

Contact the Iluvatar administrator to get the missing packages:

- mmcv-2.1.0+corex.4.3.0-cp310-cp310-linux_x86_64.whl

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
python3 export.py --weight yolov3.pt --output yolov3.onnx

# Use onnxsim optimize onnx model
onnxsim yolov3.onnx yolov3_opt.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/coco/
```

### FP16

```bash
# Accuracy
bash scripts/infer_yolov3_fp16_accuracy.sh
# Performance
bash scripts/infer_yolov3_fp16_performance.sh
```

### INT8

```bash
# Accuracy
bash scripts/infer_yolov3_int8_accuracy.sh
# Performance
bash scripts/infer_yolov3_int8_performance.sh
```

## Model Results

| Model  | BatchSize | Precision | FPS    | MAP@0.5 | MAP@0.5:0.95 |
| :----: | :----: | :----: | :----: | :----: | :----: |
| YOLOv3 | 32        | FP16      | 312.47 | 0.658   | 0.467        |
| YOLOv3 | 32        | INT8      | 711.72 | 0.639   | 0.427        |
