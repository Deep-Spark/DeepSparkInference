# FCOS (IGIE)

## Model Description

FCOS is an innovative one-stage object detection framework that abandons traditional anchor box dependency and uses a fully convolutional network for per-pixel target prediction. By introducing a centerness branch and multi-scale feature fusion, FCOS enhances detection performance while simplifying the model structure, especially in detecting small and overlapping targets. Additionally, FCOS eliminates the need for hyperparameter tuning related to anchor boxes, streamlining the model training and tuning process.

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco/fcos_r50_caffe_fpn_gn-head_1x_coco-821213aa.pth>

Dataset: <http://images.cocodataset.org/zips/val2017.zip> to download the validation dataset.

```bash
wget https://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco/fcos_r50_caffe_fpn_gn-head_1x_coco-821213aa.pth
```

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
|-------|-----------|-----------|-------|---------|--------------|
| FCOS  | 32        | FP16      | 83.09 | 0.522   | 0.339        |

## References

- [mmdetection](https://github.com/open-mmlab/mmdetection.git)
