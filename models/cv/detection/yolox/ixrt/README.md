# YoloX

## Description
YOLOX is an anchor-free version of YOLO, with a simpler design but better performance! It aims to bridge the gap between research and industrial communities.
For more details, please refer to our [report on Arxiv](https://arxiv.org/abs/2107.08430).
## Setup

### Install

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-dev

pip3 install tqdm
pip3 install onnx
pip3 install onnxsim
pip3 install tabulate
pip3 install pycocotools
pip3 install ppq
pip3 install pycuda
pip3 install protobuf==3.20.0
```

### Download
Pretrained model: <https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth>

Dataset: <http://images.cocodataset.org/zips/val2017.zip> to download the validation dataset.

### Model Conversion
```bash
# install yolox
git clone https://github.com/Megvii-BaseDetection/YOLOX.git
cd YOLOX
python3 setup.py develop

# export onnx model
python3 tools/export_onnx.py --output-name ../yolox.onnx -n yolox-m -c yolox_m.pth --batch-size 32

cd ..
```

## Inference
```bash
export DATASETS_DIR=/Path/to/coco/
```
### FP16 (ILUVATAR)

```bash
# build plugin
cd plugin && mkdir build && cd build
## link libixrt.so to lib 
ln -s /usr/local/corex/lib/python3/dist-packages/tensorrt/lib/libixrt.so /usr/local/corex/lib
## make
cmake .. -DIXRT_HOME=/usr/local/corex
make -j12
# Accuracy
bash scripts/infer_yolox_fp16_accuracy.sh
# Performance
bash scripts/infer_yolox_fp16_performance.sh
```
### INT8 (ILUVATAR)

```bash
# Accuracy
bash scripts/infer_yolox_int8_accuracy.sh
# Performance
bash scripts/infer_yolox_int8_performance.sh
```


### FP16 (NVIDIA)

```bash
# build plugin
cd plugin && mkdir build && cd build
cmake .. -DUSE_TRT=1
make -j12
# Accuracy
bash scripts/infer_yolox_fp16_accuracy.sh
# Performance
bash scripts/infer_yolox_fp16_performance.sh
```

### INT8 (NVIDIA)

```bash
# Accuracy
bash scripts/infer_yolox_int8_accuracy.sh
# Performance
bash scripts/infer_yolox_int8_performance.sh
```

## Results

Model   |BatchSize  |Precision |FPS       |MAP@0.5   |
--------|-----------|----------|----------|----------|
yolox   |    32     |   FP16   | 424.53  |  0.656   |
yolox   |    32     |   INT8   | 832.16  |  0.647   |


## Reference

YOLOX: https://github.com/Megvii-BaseDetection/YOLOX
