# Swin Transformer Large (ixRT)

## Model Description

Swin Transformer-Large is a variant of the Swin Transformer, an architecture designed for computer vision tasks, particularly within the realms of image classification, object detection, and segmentation. The Swin Transformer-Large model represents an expanded version with more layers and parameters compared to its base configuration, aiming for improved performance and deeper processing of visual data.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.09 |
| MR-V100 | 4.2.0 | 25.03 |

## Model Preparation

### Prepare Resources

Pretrained model: <https://lf-bytemlperf.17mh.cn/obj/bytemlperf-zoo/open-swin-large.tar>

Dataset: <https://lf-bytemlperf.17mh.cn/obj/bytemlperf-zoo/open_imagenet.tar> to download the open_imagenet dataset.

or you can :

```bash
bash ./scripts/prepare_model_and_dataset.sh
```

### Install Dependencies

Contact the Iluvatar administrator to get the missing packages:
- tensorflow-2.16.2+corex.4.3.0-cp310-cp310-linux_x86_64.whl

```bash
export PROJ_ROOT=/PATH/TO/DEEPSPARKINFERENCE
export MODEL_PATH=${PROJ_ROOT}/models/cv/classification/swin_transformer_large/ixrt
cd ${MODEL_PATH}

apt install -y libnuma-dev libgl1-mesa-glx

pip3 install -r requirements.txt
```

### Model Conversion

Please correct the paths in the following commands or files.

```bash
tar -xvf open-swin-large.tar
wget https://raw.githubusercontent.com/bytedance/ByteMLPerf/main/byte_infer_perf/general_perf/model_zoo/swin-large-torch-fp32.json
python3 torch2onnx.py --model_path ./general_perf/model_zoo/popular/swin-large/swin-transformer-large.pt --output_path swin-large-torch-fp32.onnx

```

## Model Inference

```bash
git clone https://gitee.com/deep-spark/iluvatar-corex-ixrt.git --depth=1
cp -r iluvatar-corex-ixrt/tools/optimizer/ ../../../../../toolbox/ByteMLPerf/byte_infer_perf/general_perf/backends/ILUVATAR/

export ORIGIN_ONNX_NAME=./swin-large-torch-fp32
export OPTIMIER_FILE=./iluvatar-corex-ixrt/tools/optimizer/optimizer.py
export PROJ_PATH=./
```

### Performance

```bash

bash ./scripts/infer_swinl_fp16_performance.sh
```

### Accuracy

If you want to evaluate the accuracy of this model, please visit here: <toolbox/ByteMLPerf/iluvatar_general_infer>, which integrates inference and training of many models under this framework, supporting the ILUVATAR backend

For detailed steps regarding this model, please refer to this document: <toolbox/ByteMLPerf/blob/iluvatar_general_infer/byte_infer_perf/general_perf/backends/ILUVATAR/README.zh_CN.md> Note: You need to modify the relevant paths in the code to your own correct paths.

```bash
# link and install requirements
ln -s ${PROJ_ROOT}/toolbox/ByteMLPerf ./
pip3 install -r ./ByteMLPerf/byte_infer_perf/general_perf/requirements.txt
pip3 install -r ./ByteMLPerf/byte_infer_perf/general_perf/backends/ILUVATAR/requirements.txt

# copy data
cp -r datasets/open_imagenet/* ByteMLPerf/byte_infer_perf/general_perf/datasets/open_imagenet/
mkdir -p ./ByteMLPerf/general_perf/model_zoo/popular/swin-large
cp general_perf/model_zoo/popular/swin-large/* ./ByteMLPerf/general_perf/model_zoo/popular/swin-large

# run acc scripts
cd ./ByteMLPerf/byte_infer_perf/general_perf
mkdir -p workloads
wget -O workloads/swin-large-torch-fp32.json https://raw.githubusercontent.com/bytedance/ByteMLPerf/refs/heads/main/byte_infer_perf/general_perf/workloads/swin-large-torch-fp32.json
python3 core/perf_engine.py --hardware_type ILUVATAR --task swin-large-torch-fp32
```

## Model Results

| Model                  | BatchSize | Precision | QPS   | Top-1 Acc |
| ---------------------- | --------- | --------- | ----- | --------- |
| Swin Transformer Large | 2         | FP16      | 5.746 | 85.62     |
