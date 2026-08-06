# YOLOv8n-pose (IGIE)

## Model Description

YOLOv8n-pose is the nano pose-estimation variant of Ultralytics YOLOv8. It jointly detects persons and predicts 17 COCO keypoints in a single forward pass, offering a compact real-time solution for human pose estimation.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 26.09 |

## Model Preparation

### Prepare Resources

Pretrained model: <https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-pose.pt>

Dataset:

- <https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels-pose.zip> to download the pose labels dataset.
- <http://images.cocodataset.org/zips/val2017.zip> to download the validation images.
- COCO person keypoints annotations: `annotations/person_keypoints_val2017.json`.

```bash
# download the weight
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-pose.pt

# prepare coco-pose (labels from zip; reuse COCO val images / annotations)
unzip -q -d /Path/to/ coco2017labels-pose.zip
# symlink images & keypoints annotation into coco-pose/
# coco-pose/
# ├── annotations/person_keypoints_val2017.json
# ├── images/val2017/
# ├── labels/val2017/
# └── val2017.txt
```

### Install Dependencies

```bash
pip3 install -r requirements.txt
```

### Model Conversion

```bash
# Export static-shape ONNX for Relay (batch=32, imgsz=640)
python3 export.py --weight yolov8n-pose.pt --batch 32
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/coco-pose/
```

### FP16

```bash
# Accuracy
bash scripts/infer_yolov8n_pose_fp16_accuracy.sh
# Performance
bash scripts/infer_yolov8n_pose_fp16_performance.sh
```

## Model Results

| Model        | BatchSize | Precision | FPS     | Box mAP@0.5 | Box mAP@0.5:0.95 | Pose mAP@0.5 | Pose mAP@0.5:0.95 |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| YOLOv8n-pose | 32        | FP16      | 2015.42 | 0.713       | 0.526            | 0.800        | 0.505             |
