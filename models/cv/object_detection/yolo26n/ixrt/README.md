# YOLO26n (ixRT)

## Model Description

YOLOv26 is the latest generation of real-time detection models released by Ultralytics in 2026. Its core evolution lies in the native NMS-Free design, enabling direct end-to-end output from model to prediction. By incorporating the MuSGD optimizer and STAL strategy, it significantly streamlines post-processing logic while further enhancing inference speed and small object detection precision.

## Supported Environments

| GPU | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release | Branch |
| :----: | :----: | :----: | :----: |
| BI-V150 | 5.0.0 | 26.12 | release/26.12 |

> **Note:** 请切换到与您的 SDK 版本对应的 Release 分支进行测试。请勿直接在 master 分支上运行测试，因为 master 分支可能包含与您的本地 SDK 版本不兼容的最新更改。
>
> 切换分支命令示例：`git checkout release/26.12`

## Model Preparation

### Prepare Resources

Pretrained model: <https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt>

Dataset:

- <http://images.cocodataset.org/zips/val2017.zip> to download the validation dataset.
- <http://images.cocodataset.org/annotations/annotations_trainval2017.zip> to download the annotations.

```bash
unzip -q -d ./coco/images/ val2017.zip
unzip -q -d ./coco/ annotations_trainval2017.zip

coco
├── annotations
│   └── instances_val2017.json
└── images
    └── val2017
```

### Install Dependencies

```bash
pip3 install --no-deps -r requirements.txt
```

> `--no-deps` 保留镜像自带的 CoreX PyTorch；不加这个参数，ultralytics 会把它替换成官方 CUDA 版本。

## Model Conversion

```bash
mkdir checkpoints
python3 export.py --weight yolo26n.pt --batch 32
mv yolo26n.onnx checkpoints/
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
bash scripts/infer_yolo26n_fp16_accuracy.sh
# Performance
bash scripts/infer_yolo26n_fp16_performance.sh
```

## Model Results

| Model | BatchSize | Precision | FPS | IOU@0.5 | IOU@0.5:0.95 |
| ------ | --------- | --------- | -------- | ------- | ------------ |
| YOLO26n | 32 | FP16 | 1784.975 | 0.5688 | 0.4088 |

> 精度在 COCO val2017 全部 5000 张图上测得，共 748052 个检测框。

## References

- [YOLO26](https://github.com/ultralytics/ultralytics)
