# YOLOv8 (IGIE)

## Model Description

YOLOv8n combines exceptional speed and competitive accuracy in real-time object detection tasks. With a focus on simplicity and efficiency, this compact model employs a single neural network to make predictions, enabling rapid and reliable identification of objects in images or video streams, making it ideal for resource-constrained environments.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.09 |
| MR-V100 | 4.2.0 | 25.03 |

## Model Preparation

### Prepare Resources

Pretrained model: <https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt>

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
python3 export.py --weight yolov8n.pt --batch 32
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/coco/
```

### FP16

```bash
# Accuracy
bash scripts/infer_yolov8n_fp16_accuracy.sh
# Performance
bash scripts/infer_yolov8n_fp16_performance.sh
```

### INT8

```bash
# Accuracy
bash scripts/infer_yolov8n_int8_accuracy.sh
# Performance
bash scripts/infer_yolov8n_int8_performance.sh
```

## Model Results

| Model  | BatchSize | Precision | FPS     | MAP@0.5 | MAP@0.5:0.95 |
| :----: | :----: | :----: | :----: | :----: | :----: |
| YOLOv8n | 32        | FP16      | 1149.667 | 0.526   | 0.373        |
| YOLOv8n | 32        | INT8      | 1212.266 | 0.503    | 0.348        |
