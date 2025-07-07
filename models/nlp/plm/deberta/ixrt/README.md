# DeBERTa (IxRT)

## Model Description

DeBERTa (Decoding-enhanced BERT with disentangled attention) is an enhanced version of the BERT (Bidirectional Encoder
Representations from Transformers) model. It improves text representation learning by introducing disentangled attention
mechanisms and decoding enhancement techniques.DeBERTa introduces disentangled attention mechanisms that decompose the
self-attention matrix into different parts, focusing on different semantic information. This helps the model better
capture relationships between texts.By incorporating decoding enhancement techniques, DeBERTa adjusts the decoder during
fine-tuning to better suit specific downstream tasks, thereby improving the model’s performance on those tasks.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.2.0     |  25.03  |

**This model is compatible with IXUCA SDK up to version 4.2.0.**

## Model Preparation

### Prepare Resources

Pretrained model: <<https://lf-bytemlperf.17mh.cn/obj/bytemlperf-zoo/open_deberta.tar> >

Dataset: <<https://lf-bytemlperf.17mh.cn/obj/bytemlperf-zoo/open_squad.tar> > to download the squad dataset.

```bash
bash ./scripts/prepare_model_and_dataset.sh
```

### Install Dependencies

```bash
export PROJ_ROOT=/PATH/TO/DEEPSPARKINFERENCE
export MODEL_PATH=${PROJ_ROOT}/models/nlp/language_model/deberta/ixrt
cd ${MODEL_PATH}

apt install -y libnuma-dev

pip3 install -r requirements.txt
```

### Model Conversion

```bash
wget https://raw.githubusercontent.com/bytedance/ByteMLPerf/main/byte_infer_perf/general_perf/model_zoo/deberta-torch-fp32.json
python3 torch2onnx.py --model_path ./general_perf/model_zoo/popular/open_deberta/deberta-base-squad.pt --output_path deberta-torch-fp32.onnx
onnxsim deberta-torch-fp32.onnx deberta-torch-fp32-sim.onnx
python3 remove_clip_and_cast.py

```

## Model Inference

```bash
git clone https://gitee.com/deep-spark/iluvatar-corex-ixrt.git --depth=1

export ORIGIN_ONNX_NAME=./deberta-sim-drop-clip-drop-invaild-cast
export OPTIMIER_FILE=./iluvatar-corex-ixrt/tools/optimizer/optimizer.py
export PROJ_PATH=./
```

### Performance

```bash

bash scripts/infer_deberta_fp16_performance.sh
```

### Accuracy

If you want to evaluate the accuracy of this model, please visit the website: <
<https://github.com/yudefu/ByteMLPerf/tree/iluvatar_general_infer> >, which integrates inference and training of many
models under this framework, supporting the ILUVATAR backend

For detailed steps regarding this model, please refer to this document: <
<https://github.com/yudefu/ByteMLPerf/blob/iluvatar_general_infer/byte_infer_perf/general_perf/backends/ILUVATAR/README.zh_CN.md>
> Note: You need to modify the relevant paths in the code to your own correct paths.

```bash
# link and install requirements
ln -s ${PROJ_ROOT}/toolbox/ByteMLPerf ./

pip3 install -r ./ByteMLPerf/byte_infer_perf/general_perf/requirements.txt
pip3 install -r ./ByteMLPerf/byte_infer_perf/general_perf/backends/ILUVATAR/requirements.txt

# setup
cp ./datasets/open_squad/* ./ByteMLPerf/byte_infer_perf/general_perf/datasets/open_squad/

mv ./deberta-sim-drop-clip-drop-invaild-cast.onnx general_perf/model_zoo/popular/open_deberta/
mv ./general_perf/model_zoo/popular/ ./ByteMLPerf/byte_infer_perf/general_perf/model_zoo/

cd ./ByteMLPerf/byte_infer_perf/general_perf
mkdir -p workloads
wget -O workloads/deberta-torch-fp32.json https://raw.githubusercontent.com/bytedance/ByteMLPerf/refs/heads/main/byte_infer_perf/general_perf/workloads/deberta-torch-fp32.json
wget http://files.deepspark.org.cn:880/deepspark/Palak.tar
tar -zxvf Palak.tar

#接着修改代码：ByteMLPerf/byte_infer_perf/general_perf/datasets/open_squad/data_loader.py -AutoTokenizer.from_pretrained("Palak/microsoft_deberta-base_squad") => AutoTokenizer.from_pretrained("/Your/Path/Palak/microsoft_deberta-base_squad")

# run acc perf
sed -i 's/tensorrt_legacy/tensorrt/g' backends/ILUVATAR/common.py
python3 core/perf_engine.py --hardware_type ILUVATAR --task deberta-torch-fp32
```

## Model Results

| Model   | BatchSize | Precision | QPS   | Exact Match | F1 Score |
| :----: | :----: | :----: | :----: | :----: | :----: |
| DeBERTa | 1         | FP16      | 18.58 | 73.76       | 81.24    |
