# FSAF (ixRT)

## Model Description

The FSAF (Feature Selective Anchor-Free) module is an innovative component for single-shot object detection that enhances performance through online feature selection and anchor-free branches. The FSAF module dynamically selects the most suitable feature level for each object instance, rather than relying on traditional anchor-based heuristic methods. This improvement significantly boosts the accuracy of object detection, especially for small targets and in complex scenes. Moreover, compared to existing anchor-based detectors, the FSAF module maintains high efficiency while adding negligible additional inference overhead.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.09 |
| MR-V100 | 4.2.0 | 25.03 |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.openmmlab.com/mmdetection/v2.0/fsaf/fsaf_r50_fpn_1x_coco/fsaf_r50_fpn_1x_coco-94ccc51f.pth>

Dataset: <http://images.cocodataset.org/zips/val2017.zip> to download the validation dataset.

```bash
wget https://download.openmmlab.com/mmdetection/v2.0/fsaf/fsaf_r50_fpn_1x_coco/fsaf_r50_fpn_1x_coco-94ccc51f.pth
```

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
| FSAF  | 32        | FP16      | 133.85 | 0.530   | 0.345        |

## References

- [mmdetection](https://github.com/open-mmlab/mmdetection.git)
