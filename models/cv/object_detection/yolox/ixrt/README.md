# YOLOX (ixRT)

## Model Description

YOLOX is an anchor-free version of YOLO, with a simpler design but better performance! It aims to bridge the gap between research and industrial communities.
For more details, please refer to our [report on Arxiv](https://arxiv.org/abs/2107.08430).

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
```

### Model Conversion

```bash
# install yolox
git clone https://github.com/Megvii-BaseDetection/YOLOX.git

pushd YOLOX
python3 setup.py develop
# export onnx model
python3 tools/export_onnx.py --output-name ../yolox.onnx -n yolox-m -c yolox_m.pth --batch-size 32
popd
```

## Model Inference

```bash
# Set DATASETS_DIR
export DATASETS_DIR=/Path/to/coco/

# Build plugin on ILUVATAR env
cd plugin && mkdir build && cd build
cmake .. -DIXRT_HOME=/usr/local/corex
make -j12

# Build plugin on NVIDIA env
cd plugin && mkdir build && cd build
cmake .. -DUSE_TRT=1
make -j12
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

| Model | BatchSize | Precision | FPS    | MAP@0.5 |
| :----: | :----: | :----: | :----: | :----: |
| YOLOX | 32        | FP16      | 424.53 | 0.656   |
| YOLOX | 32        | INT8      | 832.16 | 0.647   |

## References

- [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
