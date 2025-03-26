# SABL (IGIE)

## Model Description

SABL (Side-Aware Boundary Localization) is an innovative approach in object detection that focuses on improving the precision of bounding box localization. It addresses the limitations of traditional bounding box regression methods, such as boundary ambiguity and asymmetric prediction errors, was first proposed in the paper "Side-Aware Boundary Localization for More Precise Object Detection".

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| MR-V100 | 4.2.0     |  25.03  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.openmmlab.com/mmdetection/v2.0/sabl/sabl_retinanet_r50_fpn_1x_coco/sabl_retinanet_r50_fpn_1x_coco-6c54fd4f.pth>

Dataset: <http://images.cocodataset.org/zips/val2017.zip> to download the validation dataset.

```bash
wget https://download.openmmlab.com/mmdetection/v2.0/sabl/sabl_retinanet_r50_fpn_1x_coco/sabl_retinanet_r50_fpn_1x_coco-6c54fd4f.pth
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
python3 export.py --weight sabl_retinanet_r50_fpn_1x_coco-6c54fd4f.pth --cfg sabl-retinanet_r50_fpn_1x_coco.py --output sabl.onnx

# use onnxsim optimize onnx model
onnxsim sabl.onnx sabl_opt.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/coco/
```

### FP16

```bash
# Accuracy
bash scripts/infer_sabl_fp16_accuracy.sh
# Performance
bash scripts/infer_sabl_fp16_performance.sh
```

## Model Results

| Model | BatchSize | Precision | FPS    | IOU@0.5 | IOU@0.5:0.95 |
|-------|-----------|-----------|--------|---------|--------------|
| SABL  | 32        | FP16      | 189.42 | 0.530   | 0.356        |

## References

- [mmdetection](https://github.com/open-mmlab/mmdetection.git)
