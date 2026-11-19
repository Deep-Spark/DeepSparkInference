# YOLOv26n-sem (IGIE)

## Model Description

YOLOv26-sem is the semantic segmentation variant of Ultralytics YOLO26. It predicts a dense per-pixel class map and is pretrained on Cityscapes (19 classes), making it suitable for scene parsing tasks such as autonomous driving.

## Supported Environments

| GPU | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release | Branch |
| :----: | :----: | :----: | :----: |
| MR-V100 | 5.0.0 | 26.09 | release/26.09 |

> **Note:** 请切换到与您的 SDK 版本对应的 Release 分支进行测试。请勿直接在 master 分支上运行测试，因为 master 分支可能包含与您的本地 SDK 版本不兼容的最新更改。
>
> 切换分支命令示例：`git checkout release/26.09`

## Model Preparation

### Prepare Resources

Pretrained model: <https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-sem.pt>

Dataset: <https://www.cityscapes-dataset.com> to download Cityscapes (`leftImg8bit` + `gtFine`).

Convert to Ultralytics layout (`images/` + `masks/`):

```bash
# After extracting leftImg8bit/ and gtFine/ under cityscapes/, run Ultralytics cityscapes.yaml
# download/convert helper, or prepare:
cityscapes
├── cityscapes.yaml
├── images
│   └── val
├── masks
│   └── val
├── leftImg8bit
└── gtFine
```

`cityscapes.yaml` should set `path` to this directory, `val: images/val`, `masks_dir: masks`, and the Cityscapes 19-class `names` / `label_mapping` (see Ultralytics `cityscapes.yaml`).

### Install Dependencies

```bash
pip3 install -r requirements.txt
# make sure numpy==1.26.4
pip install numpy==1.26.4
```

## Model Conversion

```bash
# download the weight from the recommend link
wget https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-sem.pt

# Cityscapes pretrained input size is 1024x2048
python3 export.py --weight yolo26n-sem.pt --batch 32

# Use onnxsim optimize onnx model
onnxsim yolo26n-sem.onnx yolo26n-sem_opt.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/cityscapes/
```

### FP16

```bash
# Accuracy
bash scripts/infer_yolov26n_sem_fp16_accuracy.sh
# Performance
bash scripts/infer_yolov26n_sem_fp16_performance.sh
```

## Model Results

| Model        | BatchSize | Precision | FPS   | mIoU  |
| ------------ | --------- | --------- | ----- | ----- |
| YOLOv26n-sem | 32        | FP16      | 54.70 | 0.783 |

## References

- [YOLOv26 Semantic](https://docs.ultralytics.com/tasks/semantic/)
