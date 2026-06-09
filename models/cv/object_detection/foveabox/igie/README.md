# FoveaBox (IGIE)

## Model Description

FoveaBox is an advanced anchor-free object detection framework that enhances accuracy and flexibility by directly predicting the existence and bounding box coordinates of objects. Utilizing a Feature Pyramid Network (FPN), it adeptly handles targets of varying scales, particularly excelling with objects of arbitrary aspect ratios. FoveaBox also demonstrates robustness against image deformations.

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

Pretrained model: <https://download.openmmlab.com/mmdetection/v2.0/foveabox/fovea_r50_fpn_4x4_1x_coco/fovea_r50_fpn_4x4_1x_coco_20200219-ee4d5303.pth>

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

- mmcv-*.whl

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
# export onnx model
python3 export.py --weight fovea_r50_fpn_4x4_1x_coco_20200219-ee4d5303.pth --cfg fovea_r50_fpn_4xb4-1x_coco.py --output foveabox.onnx

# Use onnxsim optimize onnx model
onnxsim foveabox.onnx foveabox_opt.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/coco/
```

### FP16

```bash
# Accuracy
bash scripts/infer_foveabox_fp16_accuracy.sh
# Performance
bash scripts/infer_foveabox_fp16_performance.sh
```

## Model Results

| Model    | BatchSize | Precision | FPS     | IOU@0.5 | IOU@0.5:0.95 |
| :----: | :----: | :----: | :----: | :----: | :----: |
| FoveaBox | 32        | FP16      | 192.496 | 0.531   | 0.346        |

## References

- [mmdetection](https://github.com/open-mmlab/mmdetection.git)
