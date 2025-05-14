# ATSS (IGIE)

## Model Description

ATSS is an advanced adaptive training sample selection method that effectively enhances the performance of both anchor-based and anchor-free object detectors by dynamically choosing positive and negative samples based on the statistical characteristics of objects. The design of ATSS reduces reliance on hyperparameters, simplifies the sample selection process, and significantly improves detection accuracy without adding extra computational costs.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.2.0     |  25.03  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.openmmlab.com/mmdetection/v2.0/atss/atss_r50_fpn_1x_coco/atss_r50_fpn_1x_coco_20200209-985f7bd0.pth>

Dataset: <http://images.cocodataset.org/zips/val2017.zip> to download the validation dataset.

```bash
wget https://download.openmmlab.com/mmdetection/v2.0/atss/atss_r50_fpn_1x_coco/atss_r50_fpn_1x_coco_20200209-985f7bd0.pth
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
python3 export.py --weight atss_r50_fpn_1x_coco_20200209-985f7bd0.pth --cfg atss_r50_fpn_1x_coco.py --output atss.onnx

# use onnxsim optimize onnx model
onnxsim atss.onnx atss_opt.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/coco/
```

### FP16

```bash
# Accuracy
bash scripts/infer_atss_fp16_accuracy.sh
# Performance
bash scripts/infer_atss_fp16_performance.sh
```

## Model Results

| Model | BatchSize | Precision | FPS    | IOU@0.5 | IOU@0.5:0.95 |
|-------|-----------|-----------|--------|---------|--------------|
| ATSS  | 32        | FP16      | 81.671 | 0.541   | 0.367        |

## References

- [mmdetection](https://github.com/open-mmlab/mmdetection.git)
