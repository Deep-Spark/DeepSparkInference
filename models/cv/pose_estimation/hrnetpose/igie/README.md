# HRNetPose (IGIE)

## Model Description

HRNetPose (High-Resolution Network for Pose Estimation) is a high-performance human pose estimation model introduced in the paper "Deep High-Resolution Representation Learning for Human Pose Estimation". It is designed to address the limitations of traditional methods by maintaining high-resolution feature representations throughout the network, enabling more accurate detection of human keypoints.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.09 |
| MR-V100 | 4.2.0 | 25.03 |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth>

Dataset:
  - <https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip> to download the labels dataset.
  - <http://images.cocodataset.org/zips/val2017.zip> to download the validation dataset.
  - <http://images.cocodataset.org/zips/train2017.zip> to download the train dataset.
  - <http://images.cocodataset.org/annotations/annotations_trainval2017.zip> to download the annotations dataset.

```bash
unzip -q -d ./ coco2017labels.zip
unzip -q -d ./coco/images/ train2017.zip
unzip -q -d ./coco/images/ val2017.zip
unzip -q -d ./coco annotations_trainval2017.zip

coco
├── annotations
│   └── instances_train2017.json
│   └── instances_val2017.json
│   └── captions_train2017.json
│   └── captions_val2017.json
│   └── person_keypoints_train2017.json
│   └── person_keypoints_val2017.json
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
wget https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth
```

### Install Dependencies

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx
# before install mmpose==1.3.1 need to install chchumpy==0.70 which is too older that is not compatible with newer Python versions or pip
# so need to downgrade pip to version 20.2.4
pip install pip==20.2.4
pip install mmpose==1.3.1
pip install --upgrade pip
pip install -r requirements.txt
```

### Model Conversion

```bash
# export onnx model
python3 export.py --weight hrnet_w32_coco_256x192-c78dce93_20200708.pth --cfg td-hm_hrnet-w32_8xb64-210e_coco-256x192.py --input 1,3,256,192  --output hrnetpose.onnx

# use onnxsim optimize onnx model
onnxsim hrnetpose.onnx hrnetpose_opt.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/coco/
```

### FP16

```bash
# Accuracy
bash scripts/infer_hrnetpose_fp16_accuracy.sh
# Performance
bash scripts/infer_hrnetpose_fp16_performance.sh
```

## Model Results

| Model     | BatchSize | Input Shape | Precision | FPS     | mAP@0.5(%) |
| :----: | :----: | :----: | :----: | :----: | :----: |
| HRNetPose | 32        | 252x196     | FP16      | 1831.20 | 0.926      |

## References

- [mmpose](https://github.com/open-mmlab/mmpose.git)
