# RoBERTa

## Description

Language model pretraining has led to significant performance gains but careful comparison between different approaches is challenging. Training is computationally expensive, often done on private datasets of different sizes, and, as we will show, hyperparameter choices have significant impact on the final results. We present a replication study of BERT pretraining (Devlin et al., 2019) that carefully measures the impact of many key hyperparameters and training data size. We find that BERT was significantly undertrained, and can match or exceed the performance of every model published after it. Our best model achieves state-of-the-art results on GLUE, RACE and SQuAD. These results highlight the importance of previously overlooked design choices, and raise questions about the source of recently reported improvements. We release our models and code.

## Setup

### Install

```bash
export PROJ_ROOT=/PATH/TO/DEEPSPARKINFERENCE
export MODEL_PATH=${PROJ_ROOT}/models/nlp/language_model/roberta/ixrt
cd ${MODEL_PATH}

pip3 install onnxsim
pip3 install py-libnuma==1.2
pip3 install bert
pip3 install pycuda
pip3 install transformers==4.33.3
```

### Download

Pretrained model: <https://lf-bytemlperf.17mh.cn/obj/bytemlperf-zoo/open_roberta.tar>

Dataset: <https://lf-bytemlperf.17mh.cn/obj/bytemlperf-zoo/open_squad.tar>

```bash
# Go to path of this model
cd ${PROJ_ROOT}/models/nlp/language_model/roberta/ixrt/

# get open_roberta
wget https://lf-bytemlperf.17mh.cn/obj/bytemlperf-zoo/open_roberta.tar
tar xf open_roberta.tar
rm -f open_roberta.tar

# get roberta-torch-fp32.json
wget https://raw.githubusercontent.com/bytedance/ByteMLPerf/main/byte_infer_perf/general_perf/model_zoo/roberta-torch-fp32.json

# export onnx
python3 export_onnx.py --model_path open_roberta/roberta-base-squad.pt --output_path open_roberta/roberta-torch-fp32.onnx

# Simplify onnx model
onnxsim open_roberta/roberta-torch-fp32.onnx open_roberta/roberta-torch-fp32_sim.onnx
```

## Inference

```bash
export ORIGIN_ONNX_NAME=./open_roberta/roberta-torch-fp32_sim
export OPTIMIER_FILE=${IXRT_OSS_ROOT}/tools/optimizer/optimizer.py
export PROJ_PATH=./
```

### Performance

```bash
bash scripts/infer_roberta_fp16_performance.sh
```

### Accuracy

If you want to evaluate the accuracy of this model, please visit the website: <https://github.com/yudefu/ByteMLPerf/tree/iluvatar_general_infer>, which integrates inference and training of many models under this framework, supporting the ILUVATAR backend

For detailed steps regarding this model, please refer to this document: <https://github.com/yudefu/ByteMLPerf/blob/iluvatar_general_infer/byte_infer_perf/general_perf/backends/ILUVATAR/README.zh_CN.md> Note: You need to modify the relevant paths in the code to your own correct paths.

```bash
# Link and install requirements
ln -s ${PROJ_ROOT}/toolbox/ByteMLPerf ./
pip3 install -r ./ByteMLPerf/byte_infer_perf/general_perf/requirements.txt
pip3 install -r ./ByteMLPerf/byte_infer_perf/general_perf/backends/ILUVATAR/requirements.txt
mv perf_engine.py ./ByteMLPerf/byte_infer_perf/general_perf/core/perf_engine.py

# Move open_roberta
mkdir -p ./ByteMLPerf/byte_infer_perf/general_perf/model_zoo/popular/
mv open_roberta ./ByteMLPerf/byte_infer_perf/general_perf/model_zoo/popular/

# Get open_squad
wget https://lf-bytemlperf.17mh.cn/obj/bytemlperf-zoo/open_squad.tar
tar xf open_squad.tar
cp ./open_squad/* ./ByteMLPerf/byte_infer_perf/general_perf/datasets/open_squad
rm -f open_squad.tar

# Get csarron.tar
wget http://files.deepspark.org.cn:880/deepspark/csarron.tar
tar xf csarron.tar
rm -f csarron.tar
mv csarron/ ./ByteMLPerf/byte_infer_perf/

# Run Acc scripts
cd ./ByteMLPerf/byte_infer_perf/
python3 general_perf/core/perf_engine.py --hardware_type ILUVATAR --task roberta-torch-fp32
```

## Results

| Model   | BatchSize | Precision | FPS    | F1       | Exact Match |
| ------- | --------- | --------- | ------ | -------- | ----------- |
| RoBERTa | 1         | FP16      | 355.48 | 83.14387 | 76.50175    |
