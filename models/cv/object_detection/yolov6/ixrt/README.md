# YOLOv6 (ixRT)

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

Dataset:

- <https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip> to download the labels dataset.
- <http://images.cocodataset.org/zips/val2017.zip> to download the validation dataset.
- <http://images.cocodataset.org/zips/train2017.zip> to download the train dataset.

```bash
unzip -q -d ./ coco2017labels.zip
unzip -q -d ./coco/images/ train2017.zip
unzip -q -d ./coco/images/ val2017.zip

coco
├── annotations
│   └── instances_val2017.json
├── images
│   ├── train2017
│   └── val2017
├── labels
│   ├── train2017
│   └── val2017
├── LICENSE
├── README.txt
├── test-dev2017.txt
├── train2017.cache
├── train2017.txt
├── val2017.cache
└── val2017.txt
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

```bash
# get yolov6s.pt
wget https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6s.pt
```

### Model Conversion

```bash
mkdir -p data/
# install yolov6
git clone https://github.com/meituan/YOLOv6.git

pushd YOLOv6
pip3 install -r requirements.txt

# export onnx model
python3 deploy/ONNX/export_onnx.py --weights ../yolov6s.pt --img 640 --batch-size 32 --simplify
mv ../yolov6s.onnx ../data/

popd
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

### INT8

```bash
# Accuracy
bash scripts/infer_yolov6_int8_accuracy.sh
# Performance
bash scripts/infer_yolov6_int8_performance.sh
```

## Model Results

| Model  | BatchSize | Precision | FPS      | MAP@0.5 |
| :----: | :----: | :----: | :----: | :----: |
| YOLOv6 | 32        | FP16      | 1107.511 | 0.617   |
| YOLOv6 | 32        | INT8      | 2080.475 | 0.583   |

## References

- [YOLOv6](https://github.com/meituan/YOLOv6)
