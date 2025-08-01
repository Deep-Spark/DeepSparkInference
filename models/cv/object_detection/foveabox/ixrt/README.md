# FoveaBox (ixRT)

## Model Description

FoveaBox is an advanced anchor-free object detection framework that enhances accuracy and flexibility by directly predicting the existence and bounding box coordinates of objects. Utilizing a Feature Pyramid Network (FPN), it adeptly handles targets of varying scales, particularly excelling with objects of arbitrary aspect ratios. FoveaBox also demonstrates robustness against image deformations.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.09 |
| MR-V100 | 4.2.0 | 25.03 |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.openmmlab.com/mmdetection/v2.0/foveabox/fovea_r50_fpn_4x4_1x_coco/fovea_r50_fpn_4x4_1x_coco_20200219-ee4d5303.pth>

Dataset: <http://images.cocodataset.org/zips/val2017.zip> to download the validation dataset.

### Install Dependencies

Contact the Iluvatar administrator to get the missing packages:
- mmcv-2.1.0+corex.4.3.0-cp310-cp310-linux_x86_64.whl

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
| FoveaBox | 32        | FP16      | 181.304 | 0.531   | 0.346        |

## References

- [mmdetection](https://github.com/open-mmlab/mmdetection.git)
