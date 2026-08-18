# YOLOv26n (IGIE)

## Model Description

YOLOv26 is the latest generation of real-time detection models released by Ultralytics in 2026. Its core evolution lies in the native NMS-Free design, enabling direct end-to-end output from model to prediction. By incorporating the MuSGD optimizer and STAL strategy, it significantly streamlines post-processing logic while further enhancing inference speed and small object detection precision.

## Supported Environments

| GPU | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release | Branch |
| :----: | :----: | :----: | :----: |
| MR-V100 | 4.4.0 | 26.03 | release/26.03 |
| MR-V100 | 4.3.0 | 25.12 | release/25.12 |

> **Note:** 请切换到与您的 SDK 版本对应的 Release 分支进行测试。请勿直接在 master 分支上运行测试，因为 master 分支可能包含与您的本地 SDK 版本不兼容的最新更改。
>
> 切换分支命令示例：`git checkout release/26.03`

## Model Preparation

### Prepare Resources

Pretrained model: <https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt>

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

```bash
pip3 install -r requirements.txt
# make sure numpy==1.26.4
pip install numpy==1.26.4
```

## Model Conversion

```bash
# download the weight from the recommend link
wget https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt

python3 export.py --weight yolo26n.pt --batch 32

# Use onnxsim optimize onnx model
onnxsim yolo26n.onnx yolo26n_opt.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/coco/
```

### FP16

```bash
# Accuracy
bash scripts/infer_yolov26n_fp16_accuracy.sh
# Performance
bash scripts/infer_yolov26n_fp16_performance.sh
```

## Model Results

| Model    | BatchSize | Precision | FPS     | IOU@0.5 | IOU@0.5:0.95 |
| -------- | --------- | --------- | ------- | ------- | ------------ |
| YOLOv26n | 32        | FP16      | 1344.11 |  0.558  | 0.402        |

## References

- [YOLOv26](https://github.com/ultralytics/ultralytics)
