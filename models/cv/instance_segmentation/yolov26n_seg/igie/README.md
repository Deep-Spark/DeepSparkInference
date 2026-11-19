# YOLOv26n-seg (IGIE)

## Model Description

YOLOv26-seg is the instance segmentation variant of Ultralytics YOLO26. It retains the NMS-Free end-to-end design of YOLO26 detection while jointly predicting instance masks, delivering real-time object-level segmentation on COCO.

## Supported Environments

| GPU | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release | Branch |
| :----: | :----: | :----: | :----: |
| MR-V100 | 5.0.0 | 26.09 | release/26.09 |

> **Note:** 请切换到与您的 SDK 版本对应的 Release 分支进行测试。请勿直接在 master 分支上运行测试，因为 master 分支可能包含与您的本地 SDK 版本不兼容的最新更改。
>
> 切换分支命令示例：`git checkout release/26.09`

## Model Preparation

### Prepare Resources

Pretrained model: <https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-seg.pt>

Dataset:

- <https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels-segments.zip> to download the segmentation labels dataset.
- <http://images.cocodataset.org/zips/val2017.zip> to download the validation dataset.

```bash
unzip -q -d ./ coco2017labels-segments.zip
unzip -q -d ./coco/images/ val2017.zip

coco
├── annotations
│   └── instances_val2017.json
├── images
│   └── val2017
├── labels
│   └── val2017
└── val2017.txt
```

### Install Dependencies

```bash
pip3 install -r requirements.txt
# make sure numpy==1.26.4
pip install numpy==1.26.4
```

## Model Conversion

```bash
# download the weight from the recommend link
wget https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-seg.pt

python3 export.py --weight yolo26n-seg.pt --batch 32

# Use onnxsim optimize onnx model
onnxsim yolo26n-seg.onnx yolo26n-seg_opt.onnx
```

## Model Inference

```bash
# COCO root with segment labels under labels/val2017/
export DATASETS_DIR=/Path/to/coco/
```

### FP16

```bash
# Accuracy
bash scripts/infer_yolov26n_seg_fp16_accuracy.sh
# Performance
bash scripts/infer_yolov26n_seg_fp16_performance.sh
```

## Model Results

| Model        | BatchSize | Precision | FPS     | Box mAP@0.5 | Box mAP@0.5:0.95 | Mask mAP@0.5 | Mask mAP@0.5:0.95 |
| ------------ | --------- | --------- | ------- | ----------- | ---------------- | ------------ | ----------------- |
| YOLOv26n-seg | 32        | FP16      | 1505.87 | 0.561       | 0.400            | 0.534        | 0.336             |

## References

- [YOLOv26 Segment](https://docs.ultralytics.com/tasks/segment/)
