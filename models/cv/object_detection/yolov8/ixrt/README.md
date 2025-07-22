# YOLOv8 (IxRT)

## Model Description

Yolov8 combines speed and accuracy in real-time object detection tasks. With a focus on simplicity and efficiency, this model employs a single neural network to make predictions, enabling fast and accurate identification of objects in images or video streams.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.09 |
| MR-V100 | 4.2.0 | 25.03 |

## Model Preparation

### Prepare Resources

Pretrained model: <https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt>

Dataset: <http://images.cocodataset.org/zips/val2017.zip> to download the validation dataset.

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
mkdir -p checkpoints/
mv yolov8n.pt yolov8.pt
python3 export.py --weight yolov8.pt --batch 32
onnxsim yolov8.onnx ./checkpoints/yolov8.onnx
```

## Model Inference

```bash
export PROJ_DIR=./
export DATASETS_DIR=/path/to/coco/
export CHECKPOINTS_DIR=./checkpoints
export RUN_DIR=./
```

### FP16

```bash
# Accuracy
bash scripts/infer_yolov8_fp16_accuracy.sh
# Performance
bash scripts/infer_yolov8_fp16_performance.sh
```

### INT8

```bash
# Accuracy
bash scripts/infer_yolov8_int8_accuracy.sh
# Performance
bash scripts/infer_yolov8_int8_performance.sh
```

## Model Results

| Model  | BatchSize | Precision | FPS      | MAP@0.5 |
| :----: | :----: | :----: | :----: | :----: |
| YOLOv8 | 32        | FP16      | 1511.366 | 0.525   |
| YOLOv8 | 32        | INT8      | 1841.017 | 0.517   |
