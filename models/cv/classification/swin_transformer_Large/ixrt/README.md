# Swin-L

## Description

Swin Transformer-Large is a variant of the Swin Transformer, an architecture designed for computer vision tasks, particularly within the realms of image classification, object detection, and segmentation. The Swin Transformer-Large model represents an expanded version with more layers and parameters compared to its base configuration, aiming for improved performance and deeper processing of visual data.

## Setup

### Install

```bash
pip3 install onnxsim
pip3 install onnx_graphsurgeon
pip3 install scikit-learn
pip3 install tqdm
pip3 install pycuda
pip3 install onnx
pip3 install tabulate
pip3 install cv2
pip3 install pycocotools
pip3 install opencv-python==4.6.0.66
```

### Download

Pretrained model: <https://lf-bytemlperf.17mh.cn/obj/bytemlperf-zoo/open-swin-large.tar  >

Dataset: <https://lf-bytemlperf.17mh.cn/obj/bytemlperf-zoo/open_imagenet.tar > to download the open_imagenet dataset.

or you can :
```bash
bash /scripts/prepare_model_and_dataset.sh

```

### Model Conversion
Please correct the paths in the following commands or files.
```bash
tar -xvf open-swin-large.tar
wget < https://github.com/bytedance/ByteMLPerf/blob/main/byte_infer_perf/general_perf/model_zoo/swin-large-torch-fp32.json >
python3 torch2onnx.py --model_path swin-transformer-large.pt --output_path swin-large-torch-fp32.onnx

```

## Inference


```bash
export ORIGIN_ONNX_NAME=/Path/swin-large-torch-fp32.onnx
export OPTIMIER_FILE=/Path/ixrt/oss/tools/optimizer/optimizer.py
export PROJ_PATH=./
```

### Performance

```bash

bash scripts/infer_swinl_fp16_performance.sh
```

### Accuracy

If you want to evaluate the accuracy of this model, please visit the website: < https://github.com/yudefu/ByteMLPerf/tree/iluvatar_general_infer >, which integrates inference and training of many models under this framework, supporting the ILUVATAR backend

```bash

git clone https://github.com/yudefu/ByteMLPerf.git -b iluvatar_general_infer
```

For detailed steps regarding this model, please refer to this document: < https://github.com/yudefu/ByteMLPerf/blob/iluvatar_general_infer/byte_infer_perf/general_perf/backends/ILUVATAR/README.zh_CN.md > Note: You need to modify the relevant paths in the code to your own correct paths.

```bash

pip3 install -r https://github.com/yudefu/ByteMLPerf/blob/iluvatar_general_infer/byte_infer_perf/general_perf/requirements.txt
mv /ixrt/perf_engine.py /ByteMLPerf/byte_infer_perf/general_perf/core/perf_engine.py
cd /ByteMLPerf/byte_infer_perf/
mv /general_perf/general_perf/model_zoo/popular/swin-large /general_perf/model_zoo/popular/swin-large
cd /ByteMLPerf/byte_infer_perf/general_perf
python3 core/perf_engine.py --hardware_type ILUVATAR --task swin-large-torch-fp32
```


## Results

Model   |BatchSize  |Precision |QPS       |Top-1 Acc  |
--------|-----------|----------|----------|-----------|
Swin-L  |    16     |   FP16   | 5.746    | 85.62     | 