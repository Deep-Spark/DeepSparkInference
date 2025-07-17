# YOLOF (IGIE)

## Model Description

YOLOF is a lightweight object detection model that focuses on single-level feature maps for detection and enhances feature representation using dilated convolution modules. With a simple and efficient structure, it is well-suited for real-time object detection tasks.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0     |  25.09  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.openmmlab.com/mmdetection/v2.0/yolof/yolof_r50_c5_8x8_1x_coco/yolof_r50_c5_8x8_1x_coco_20210425_024427-8e864411.pth>

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
python3 export.py --weight yolof_r50_c5_8x8_1x_coco_20210425_024427-8e864411.pth --cfg yolof_r50-c5_8xb8-1x_coco.py --output yolof.onnx

# use onnxsim optimize onnx model
onnxsim yolof.onnx yolof_opt.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/coco/
```

### FP16

```bash
# Accuracy
bash scripts/infer_yolof_fp16_accuracy.sh
# Performance
bash scripts/infer_yolof_fp16_performance.sh
```

## Model Results

| Model | BatchSize | Precision | FPS    | IOU@0.5 | IOU@0.5:0.95 |
| :----:| :-------: | :-------: | :----: | :-----: | :----------: |
| YOLOF | 32        | FP16      | 333.59 | 0.527   | 0.343        |

## References

- [mmdetection](https://github.com/open-mmlab/mmdetection.git)
