# FSAF (IGIE)

## Model Description

The FSAF (Feature Selective Anchor-Free) module is an innovative component for single-shot object detection that enhances performance through online feature selection and anchor-free branches. The FSAF module dynamically selects the most suitable feature level for each object instance, rather than relying on traditional anchor-based heuristic methods. This improvement significantly boosts the accuracy of object detection, especially for small targets and in complex scenes. Moreover, compared to existing anchor-based detectors, the FSAF module maintains high efficiency while adding negligible additional inference overhead.

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

Pretrained model: <https://download.openmmlab.com/mmdetection/v2.0/fsaf/fsaf_r50_fpn_1x_coco/fsaf_r50_fpn_1x_coco-94ccc51f.pth>

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
wget https://download.openmmlab.com/mmdetection/v2.0/fsaf/fsaf_r50_fpn_1x_coco/fsaf_r50_fpn_1x_coco-94ccc51f.pth
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
python3 export.py --weight fsaf_r50_fpn_1x_coco-94ccc51f.pth --cfg fsaf_r50_fpn_1x_coco.py --output fsaf.onnx

# use onnxsim optimize onnx model
onnxsim fsaf.onnx fsaf_opt.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/coco/
```

### FP16

```bash
# Accuracy
bash scripts/infer_fsaf_fp16_accuracy.sh
# Performance
bash scripts/infer_fsaf_fp16_performance.sh
```

## Model Results

| Model | BatchSize | Precision | FPS    | IOU@0.5 | IOU@0.5:0.95 |
| :----: | :----: | :----: | :----: | :----: | :----: |
| FSAF  | 32        | FP16      | 122.35 | 0.530   | 0.345        |

## References

- [mmdetection](https://github.com/open-mmlab/mmdetection.git)
