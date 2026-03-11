# DETR (IGIE)

## Model Description

DETR (DEtection TRansformer) is a novel approach that views object detection as a direct set prediction problem. This method streamlines the detection process, eliminating the need for many hand-designed components like non-maximum suppression procedures or anchor generation, which are typically used to explicitly encode prior knowledge about the task.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.4.0 | 26.03 |


## Model Preparation

this is a fork version of official detr https://github.com/facebookresearch/detr.git

### Prepare Resources

Pretrained model: <https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth>

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
python3 export.py --no_aux_loss --eval --resume detr-r50-e632da11.pth --coco_path /path/data/coco

onnxsim detr.onnx detr_opt.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/path/to/coco/
```

### FP16

```bash
# Accuracy
bash scripts/infer_detr_fp16_accuracy.sh
# Performance
bash scripts/infer_detr_fp16_performance.sh
```

## Model Results

| Model | BatchSize | Precision | FPS   | MAP@0.5 | MAP@0.5:0.95 |
| :----: | :----: | :----: | :----: | :----: | :----: |
| DETR  | 32         | FP16      | 149.37 | 0.581   | 0.385        |
