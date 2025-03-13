# VideoBERT (IxRT)

## Model Description

VideoBERT is a model designed for video understanding tasks, extending the capabilities of BERT (Bidirectional Encoder
Representations from Transformers) to video data. It enhances video representation learning by integrating both visual
and textual information into a unified framework.

## Supported Environments

| Iluvatar GPU | IXUCA SDK |
|--------------|-----------|
| MR-V100      | 4.2.0     |

## Model Preparation

### Prepare Resources

Pretrained model: <https://lf-bytemlperf.17mh.cn/obj/bytemlperf-zoo/open_videobert.tar>

Dataset: <https://lf-bytemlperf.17mh.cn/obj/bytemlperf-zoo/cifar-100-python.tar> to download the cifar-100-python dataset.

or you can :

```bash
export PROJ_ROOT=/PATH/TO/DEEPSPARKINFERENCE
export MODEL_PATH=${PROJ_ROOT}/models/nlp/language_model/videobert/ixrt
cd ${MODEL_PATH}
bash ./scripts/prepare_model_and_dataset.sh
```

### Install Dependencies

```bash
apt install -y libnuma-dev

pip3 install -r requirements.txt
```

## Model Inference

```bash
git clone https://gitee.com/deep-spark/iluvatar-corex-ixrt.git --depth=1

export ORIGIN_ONNX_NAME=./general_perf/model_zoo/popular/open_videobert/video-bert
export OPTIMIER_FILE=./iluvatar-corex-ixrt/tools/optimizer/optimizer.py
export PROJ_PATH=./
```

### Performance

```bash
bash scripts/infer_videobert_fp16_performance.sh
```

### Accuracy

If you want to evaluate the accuracy of this model, please visit here: <toolbox/ByteMLPerf/byte_infer_perf/general_perf>, which integrates inference and training of many models under this framework, supporting the ILUVATAR backend

For detailed steps regarding this model, please refer to this document: <toolbox/ByteMLPerf/byte_infer_perf/general_perf/backends/ILUVATAR/README.zh_CN.md> Note: You need to modify the relevant paths in the code to your own correct paths.

```bash
# link and install requirements
ln -s ${PROJ_ROOT}/toolbox/ByteMLPerf ./
pip3 install -r ./ByteMLPerf/byte_infer_perf/general_perf/requirements.txt
pip3 install -r ./ByteMLPerf/byte_infer_perf/general_perf/backends/ILUVATAR/requirements.txt

# copy data
mkdir -p ./ByteMLPerf/byte_infer_perf/general_perf/datasets/open_cifar/
cp -r ./datasets/open_cifar/cifar-100-python/ ./ByteMLPerf/byte_infer_perf/general_perf/datasets/open_cifar/
mkdir -p ./ByteMLPerf/byte_infer_perf/general_perf/model_zoo/popular/open_videobert/
cp ./general_perf/model_zoo/popular/open_videobert/video-bert.onnx ByteMLPerf/byte_infer_perf/general_perf/model_zoo/popular/open_videobert/

# run acc scripts
cd ./ByteMLPerf/byte_infer_perf/general_perf
mkdir -p workloads
wget -O workloads/videobert-onnx-fp32.json https://raw.githubusercontent.com/bytedance/ByteMLPerf/refs/heads/main/byte_infer_perf/general_perf/workloads/videobert-onnx-fp32.json
python3 core/perf_engine.py --hardware_type ILUVATAR --task videobert-onnx-fp32
```

## Model Results

| Model     | BatchSize | Precision | QPS   | Top-1 ACC |
|-----------|-----------|-----------|-------|-----------|
| VideoBERT | 4         | FP16      | 37.68 | 61.67     |
