# FCOS (IGIE)

## Model Description

FCOS is an innovative one-stage object detection framework that abandons traditional anchor box dependency and uses a fully convolutional network for per-pixel target prediction. By introducing a centerness branch and multi-scale feature fusion, FCOS enhances detection performance while simplifying the model structure, especially in detecting small and overlapping targets. Additionally, FCOS eliminates the need for hyperparameter tuning related to anchor boxes, streamlining the model training and tuning process.

## Supported Environments

| GPU | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release | Branch |
| :----: | :----: | :----: | :----: |
| MR-V100 | 5.0.0 | 26.09 | release/26.09 |
| MR-V100 | 4.4.0 | 26.06 | release/26.06 |

> **Note:** 请切换到与您的 SDK 版本对应的 Release 分支进行测试。请勿直接在 master 分支上运行测试，因为 master 分支可能包含与您的本地 SDK 版本不兼容的最新更改。
>
> 切换分支命令示例：`git checkout release/26.09`

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco/fcos_r50_caffe_fpn_gn-head_1x_coco-821213aa.pth>

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

```bash
wget https://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco/fcos_r50_caffe_fpn_gn-head_1x_coco-821213aa.pth
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
python3 export.py --weight fcos_r50_caffe_fpn_gn-head_1x_coco-821213aa.pth --cfg fcos_r50_caffe_fpn_gn-head_1x_coco.py --output fcos.onnx

# use onnxsim optimize onnx model
onnxsim fcos.onnx fcos_opt.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/coco/
```

### FP16

```bash
# Accuracy
bash scripts/infer_fcos_fp16_accuracy.sh
# Performance
bash scripts/infer_fcos_fp16_performance.sh
```

## Model Results

| Model | BatchSize | Precision | FPS   | IOU@0.5 | IOU@0.5:0.95 |
| :----: | :----: | :----: | :----: | :----: | :----: |
| FCOS  | 32        | FP16      | 83.09 | 0.522   | 0.339        |

## References

- [mmdetection](https://github.com/open-mmlab/mmdetection.git)
