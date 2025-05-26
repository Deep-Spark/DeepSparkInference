# CenterNet (IGIE)

## Model Description

CenterNet is an efficient object detection model that simplifies the traditional object detection process by representing targets as the center points of their bounding boxes and using keypoint estimation techniques to locate these points. This model not only excels in speed, achieving real-time detection while maintaining high accuracy, but also exhibits good versatility, easily extending to tasks such as 3D object detection and human pose estimation. CenterNet's network architecture employs various optimized fully convolutional networks and combines effective loss functions, making the model training and inference process more efficient.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.2.0     |  25.03  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.openmmlab.com/mmdetection/v2.0/centernet/centernet_resnet18_140e_coco/centernet_resnet18_140e_coco_20210705_093630-bb5b3bf7.pth>

Dataset: <http://images.cocodataset.org/zips/val2017.zip> to download the validation dataset.

### Install Dependencies

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
python3 export.py --weight centernet_resnet18_140e_coco_20210705_093630-bb5b3bf7.pth --cfg centernet_r18_8xb16-crop512-140e_coco.py --output centernet.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/coco/
```

### FP16

```bash
# Accuracy
bash scripts/infer_centernet_fp16_accuracy.sh
# Performance
bash scripts/infer_centernet_fp16_performance.sh
```

## Model Results

| Model     | BatchSize | Precision | FPS    | IOU@0.5 | IOU@0.5:0.95 |
| :----: | :----: | :----: | :----: | :----: | :----: |
| CenterNet | 32        | FP16      | 799.70 | 0.423   | 0.258        |

## References

- [mmdetection](https://github.com/open-mmlab/mmdetection.git)
