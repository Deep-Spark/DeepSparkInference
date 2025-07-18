# SSD (IGIE)

## Model Description

SSD is a single convolutional neural network-based object detection model that utilizes multi-scale feature maps and default box designs to perform classification and regression simultaneously. With an efficient structure, it achieves real-time object detection and is suitable for handling objects of various sizes.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0     |  25.09  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.openmmlab.com/mmdetection/v2.0/ssd/ssd300_coco/ssd300_coco_20210803_015428-d231a06e.pth>

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
python3 export.py --weight ssd300_coco_20210803_015428-d231a06e.pth --cfg ssd300_coco.py --output ssd.onnx

# use onnxsim optimize onnx model
onnxsim ssd.onnx ssd_opt.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/coco/
```

### FP16

```bash
# Accuracy
bash scripts/infer_ssd_fp16_accuracy.sh
# Performance
bash scripts/infer_ssd_fp16_performance.sh
```

## Model Results

| Model | BatchSize | Precision | FPS    | IOU@0.5 | IOU@0.5:0.95 |
| :----:| :-------: | :-------: | :----: | :-----: | :----------: |
| SSD   | 32        | FP16      | 783.83 | 0.436   | 0.255        |

## References

- [mmdetection](https://github.com/open-mmlab/mmdetection.git)
