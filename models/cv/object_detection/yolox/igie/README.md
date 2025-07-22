# YOLOX (IGIE)

## Model Description

YOLOX is an anchor-free version of YOLO, with a simpler design but better performance! It aims to bridge the gap between research and industrial communities.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.09 |
| MR-V100 | 4.2.0 | 25.03 |

## Model Preparation

### Prepare Resources

Pretrained model: <https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth>

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

#update gcc version
source /opt/rh/devtoolset-7/enable
```

### Model Conversion

```bash
# install yolox
git clone https://github.com/Megvii-BaseDetection/YOLOX.git
cd YOLOX/
python3 setup.py develop

# export onnx model
python3 tools/export_onnx.py -c ../yolox_m.pth -o 13 -n yolox-m --input input --output output --dynamic --output-name ../yolox.onnx

cd ../
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/coco/
```

### FP16

```bash
# Accuracy
bash scripts/infer_yolox_fp16_accuracy.sh
# Performance
bash scripts/infer_yolox_fp16_performance.sh
```

### INT8

```bash
# Accuracy
bash scripts/infer_yolox_int8_accuracy.sh
# Performance
bash scripts/infer_yolox_int8_performance.sh
```

## Model Results

| Model | BatchSize | Precision | FPS     | MAP@0.5 |
| :----: | :----: | :----: | :----: | :----: |
| YOLOX | 32        | FP16      | 409.517 | 0.656   |
| YOLOX | 32        | INT8      | 844.991 | 0.637   |

## References

- [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
