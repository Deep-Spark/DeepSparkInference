# YOLOv7 (IGIE)

## Model Description

YOLOv7 is a state-of-the-art real-time object detector that surpasses all known object detectors in both speed and accuracy in the range from 5 FPS to 160 FPS.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.09 |
| MR-V100 | 4.2.0 | 25.03 |

## Model Preparation

### Prepare Resources

Pretrained model: <https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt>

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

### Model Conversion

```bash
# clone yolov7
git clone https://github.com/WongKinYiu/yolov7.git
cd yolov7
git checkout a207844b1ce82d204ab36d87d496728d3d2348e7
# set weights_only=False to be comaptible with pytorch 2.7
sed -i '252 s/map_location)/map_location, weights_only=False)/' ./models/experimental.py
# export onnx model
python3 export.py --weights ../yolov7.pt --simplify --img-size 640 640 --dynamic-batch --grid

cd ..
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/coco/
```

### FP16

```bash
# Accuracy
bash scripts/infer_yolov7_fp16_accuracy.sh
# Performance
bash scripts/infer_yolov7_fp16_performance.sh
```

### INT8

```bash
# Accuracy
bash scripts/infer_yolov7_int8_accuracy.sh
# Performance
bash scripts/infer_yolov7_int8_performance.sh
```

## Model Results

| Model  | BatchSize | Precision | FPS     | MAP@0.5 | MAP@0.5:0.95 |
| :----: | :----: | :----: | :----: | :----: | :----: |
| YOLOv7 | 32        | FP16      | 341.681 | 0.695   | 0.509        |
| YOLOv7 | 32        | INT8      | 669.783 | 0.685   | 0.473        |

## References

- [YOLOv7](https://github.com/WongKinYiu/yolov7)
