# GFL (IGIE)

## Model Description

GFL (Generalized Focal Loss) is an object detection model that utilizes an improved focal loss function to address the class imbalance problem, enhancing classification capability and improving the detection accuracy of multi-scale objects and the precision of bounding box predictions. It is suitable for object detection tasks in complex scenes.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.2.0     |  25.06  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.openmmlab.com/mmdetection/v2.0/gfl/gfl_r50_fpn_1x_coco/gfl_r50_fpn_1x_coco_20200629_121244-25944287.pth>

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
python3 export.py --weight gfl_r50_fpn_1x_coco_20200629_121244-25944287.pth --cfg gfl_r50_fpn_1x_coco.py --output gfl.onnx

# use onnxsim optimize onnx model
onnxsim gfl.onnx gfl_opt.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/coco/
```

### FP16

```bash
# Accuracy
bash scripts/infer_gfl_fp16_accuracy.sh
# Performance
bash scripts/infer_gfl_fp16_performance.sh
```

## Model Results

| Model | BatchSize | Precision | FPS    | IOU@0.5 | IOU@0.5:0.95 |
|-------|-----------|-----------|--------|---------|--------------|
| GFL   | 32        | FP16      | 139.78 | 0.552   | 0.378        |

## References

- [mmdetection](https://github.com/open-mmlab/mmdetection.git)
