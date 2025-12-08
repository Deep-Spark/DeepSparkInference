# YOLOv10 (IGIE)

## Model Description

YOLOv10, built on the Ultralytics Python package by researchers at Tsinghua University, introduces a new approach to real-time object detection, addressing both the post-processing and model architecture deficiencies found in previous YOLO versions. By eliminating non-maximum suppression (NMS) and optimizing various model components, YOLOv10 achieves state-of-the-art performance with significantly reduced computational overhead. Extensive experiments demonstrate its superior accuracy-latency trade-offs across multiple model scales.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.09 |
| MR-V100 | 4.2.0 | 25.03 |

## Model Preparation

### Prepare Resources

Pretrained model: <https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10s.pt>

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
pip3 install -r requirements.txt
pip3 install mmcv-2.1.0+corex.4.3.0-cp310-cp310-linux_x86_64.whl
```

## Model Conversion

```bash
git clone --depth 1 https://github.com/THU-MIG/yolov10.git
cd yolov10/
```

```python
# 修改如下
--- a/ultralytics/engine/exporter.py
+++ b/ultralytics/engine/exporter.py
@@ -373,6 +373,7 @@ class Exporter:
             elif isinstance(self.model, DetectionModel):
                 dynamic["output0"] = {0: "batch", 2: "anchors"}  # shape(1, 84,
 8400)

+        dynamic = {'images': {0: 'batch'}, 'output0': {0: 'batch'}}
         torch.onnx.export(
             self.model.cpu() if dynamic else self.model,  # dynamic=True only c
ompatible with cpu
```

```bash
pip3 install -e . --no-deps
cd ../

python3 export.py --weight yolov10s.pt --batch 32
# Make sure numpy < 2.0
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/coco/
```

### FP16

```bash
# Accuracy
bash scripts/infer_yolov10_fp16_accuracy.sh
# Performance
bash scripts/infer_yolov10_fp16_performance.sh
```

### INT8

```bash
# Accuracy
bash scripts/infer_yolov10_int8_accuracy.sh
# Performance
bash scripts/infer_yolov10_int8_performance.sh
```

## Model Results

| Model   | BatchSize | Precision | FPS    | IOU@0.5 | IOU@0.5:0.95 |
| ------- | --------- | --------- | ------ | ------- | ------------ |
| YOLOv10 | 32        | FP16      | 528.685 | 0.629   | 0.461        |
| YOLOv10 | 32        | INT8      | 599.318 | 0.618   | 0.444        |

## References

- [YOLOv10](https://docs.ultralytics.com/models/yolov10)
