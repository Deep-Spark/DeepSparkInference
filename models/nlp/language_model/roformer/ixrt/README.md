# RoFormer

## Description

Position encoding recently has shown effective in the transformer architecture. It enables valuable supervision for dependency modeling between elements at different positions of the sequence. In this paper, we first investigate various methods to integrate positional information into the learning process of transformer-based language models. Then, we propose a novel method named Rotary Position Embedding(RoPE) to effectively leverage the positional information. Specifically, the proposed RoPE encodes the absolute position with a rotation matrix and meanwhile incorporates the explicit relative position dependency in self-attention formulation. Notably, RoPE enables valuable properties, including the flexibility of sequence length, decaying inter-token dependency with increasing relative distances, and the capability of equipping the linear self-attention with relative position encoding. Finally, we evaluate the enhanced transformer with rotary position embedding, also called RoFormer, on various long text classification benchmark datasets.

## Setup

### Install

```bash
pip3 install tf2onnx
pip3 install pycuda
pip3 install onnxsim
pip3 install py-libnuma==1.2

```

### Download

Pretrained model: <https://lf-bytemlperf.17mh.cn/obj/bytemlperf-zoo/open_roformer.tar>

Dataset: <https://lf-bytemlperf.17mh.cn/obj/bytemlperf-zoo/open_cail2019.tar>

```bash
# Download the pretrained model and dataset to 'data'
mkdir data

# export onnx
python3 export_onnx.py --model_path ./data/open_roformer --output_path ./data/open_roformer/roformer-frozen_org.onnx

# Simplify onnx model
onnxsim ./data/open_roformer/roformer-frozen_org.onnx ./data/open_roformer/roformer-frozen.onnx
python3 deploy.py --model_path ./data/open_roformer/roformer-frozen.onnx --output_path ./data/open_roformer/roformer-frozen.onnx

```

## Inference

```bash
export ORIGIN_ONNX_NAME=/Path/roformer-frozen
export OPTIMIER_FILE=/Path/ixrt/oss/tools/optimizer/optimizer.py
export PROJ_PATH=./
```

### Performance

```bash
bash scripts/infer_roformer_fp16_performance.sh
```

### Accuracy

If you want to evaluate the accuracy of this model, please visit the website: < https://github.com/yudefu/ByteMLPerf/tree/iluvatar_general_infer >, which integrates inference and training of many models under this framework, supporting the ILUVATAR backend

```bash

git clone https://github.com/yudefu/ByteMLPerf.git -b iluvatar_general_infer
```

For detailed steps regarding this model, please refer to this document: < https://github.com/yudefu/ByteMLPerf/blob/iluvatar_general_infer/byte_infer_perf/general_perf/backends/ILUVATAR/README.zh_CN.md > Note: You need to modify the relevant paths in the code to your own correct paths.

```bash

pip3 install -r https://github.com/yudefu/ByteMLPerf/blob/iluvatar_general_infer/byte_infer_perf/general_perf/requirements.txt
mv perf_engine.py ./ByteMLPerf/byte_infer_perf/general_perf/core/perf_engine.py

mkdir -p ./ByteMLPerf/byte_infer_perf/general_perf/model_zoo/popular/
# Delete Line102 ' build_engine(model_name=model_name, onnx_model_path=onnx_model_path, engine_path=engine_path, MaxBatchSize=MaxBatchSize, BuildFlag='FP16') ' which is the build engine process of conformer in the file ./ByteMLPerf/byte_infer_perf/general_perf/backends/ILUVATAR/compile_backend_iluvatar.py
mv ./data/open_roformer ./ByteMLPerf/byte_infer_perf/general_perf/model_zoo/popular/
# Make sure the roformer-frozen_end.onnx is in the path "./data/open_roformer". Or you should move it to './ByteMLPerf/byte_infer_perf/general_perf/model_zoo/popular/open_roformer/'.
# mv path/to/roformer-frozen_end.onnx ./ByteMLPerf/byte_infer_perf/general_perf/model_zoo/popular/open_roformer/
wget https://lf-bytemlperf.17mh.cn/obj/bytemlperf-zoo/open_cail2019.tar -P ./ByteMLPerf/byte_infer_perf/general_perf/datasets/open_cail2019
cd ./ByteMLPerf/byte_infer_perf/general_perf
# Modify model_zoo/roformer-tf-fp32.json
# "inputs": "input_segment:0,input_token:0" --> "inputs": "input_segment0,input_token0"
# "input_shape": {"input_segment:0": [1, 1024], "input_token:0": [1, 1024]} -->"input_shape": {"input_segment0": [1, 1024], "input_token0": [1, 1024]}
python3 core/perf_engine.py --hardware_type ILUVATAR --task roformer-tf-fp32
```