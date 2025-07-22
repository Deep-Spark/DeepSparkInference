# YOLOv6 (IGIE)

## Model Description

YOLOv6 integrates cutting-edge object detection advancements from industry and academia, incorporating recent innovations in network design, training strategies, testing techniques, quantization, and optimization methods. This culmination results in a suite of deployment-ready networks, accommodating varied use cases across different scales.  

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.09 |
| MR-V100 | 4.2.0 | 25.03 |

## Model Preparation

### Prepare Resources

Pretrained model: <https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6s.pt>

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
# install yolov6
git clone https://github.com/meituan/YOLOv6.git
cd YOLOv6
pip3 install -r requirements.txt

# export onnx model
python3 deploy/ONNX/export_onnx.py --weights ../yolov6s.pt --img 640 --dynamic-batch --simplify

cd ..
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/coco/
```

### FP16

```bash
# Accuracy
bash scripts/infer_yolov6_fp16_accuracy.sh
# Performance
bash scripts/infer_yolov6_fp16_performance.sh
```

## Model Results

| Model  | BatchSize | Precision | FPS     | MAP@0.5 | MAP@0.5:0.95 |
| :----: | :----: | :----: | :----: | :----: | :----: |
| YOLOv6 | 32        | FP16      | 994.902 | 0.617   | 0.448        |

## References

- [YOLOv6](https://github.com/meituan/YOLOv6)
