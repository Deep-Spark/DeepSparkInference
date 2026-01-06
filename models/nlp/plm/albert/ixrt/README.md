# ALBERT (ixRT)

## Model Description

Albert (A Lite BERT) is a variant of the BERT (Bidirectional Encoder Representations from Transformers) model that focuses on efficiency and scalability while maintaining strong performance in natural language processing tasks. The AlBERT model introduces parameter reduction techniques and incorporates self-training strategies to enhance its effectiveness.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.09 |
| MR-V100 | 4.2.0 | 25.03 |

## Model Preparation

### Prepare Resources

Pretrained model: <https://lf-bytemlperf.17mh.cn/obj/bytemlperf-zoo/open_albert.tar>

Dataset: <https://lf-bytemlperf.17mh.cn/obj/bytemlperf-zoo/open_squad.tar> to download the squad dataset.

or you can :

```bash
export PROJ_ROOT=/PATH/TO/DEEPSPARKINFERENCE
export MODEL_PATH=${PROJ_ROOT}/models/nlp/language_model/albert/ixrt
cd ${MODEL_PATH}
bash ./scripts/prepare_model_and_dataset.sh
```

### Install Dependencies

Contact the Iluvatar administrator to get the missing packages:
- tensorflow-*.whl
- ixrt-*.whl
- cuda_python-*.whl

```bash
apt install -y libnuma-dev

pip3 install -r requirements.txt
```

### Model Conversion

Please correct the paths in the following commands or files.

```bash
wget https://raw.githubusercontent.com/bytedance/ByteMLPerf/main/byte_infer_perf/general_perf/model_zoo/albert-torch-fp32.json
python3 torch2onnx.py --model_path ./general_perf/model_zoo/popular/open_albert/albert-base-squad.pt --output_path albert-torch-fp32.onnx
onnxsim albert-torch-fp32.onnx albert-torch-fp32-sim.onnx
```

## Model Inference

```bash
git clone https://gitee.com/deep-spark/iluvatar-corex-ixrt.git --depth=1
cp -r iluvatar-corex-ixrt/tools/optimizer/ ../../../../../toolbox/ByteMLPerf/byte_infer_perf/general_perf/backends/ILUVATAR/

export ORIGIN_ONNX_NAME=./albert-torch-fp32-sim
export OPTIMIER_FILE=./iluvatar-corex-ixrt/tools/optimizer/optimizer.py
export PROJ_PATH=./
```

### Performance

```bash
bash scripts/infer_albert_fp16_performance.sh
```

### Accuracy

```bash
# get madlag.tar
wget http://files.deepspark.org.cn:880/deepspark/madlag.tar
tar xvf madlag.tar
rm -f madlag.tar

# link and install requirements
ln -s ${PROJ_ROOT}/toolbox/ByteMLPerf ./
pip3 install -r ./ByteMLPerf/byte_infer_perf/general_perf/requirements.txt
pip3 install -r ./ByteMLPerf/byte_infer_perf/general_perf/backends/ILUVATAR/requirements.txt

# edit madlag/albert-base-v2-squad path
sed -i "s#madlag#/${MODEL_PATH}/madlag#" ./ByteMLPerf/byte_infer_perf/general_perf/datasets/open_squad/data_loader.py

# copy open_squad data
cp datasets/open_squad/* ./ByteMLPerf/byte_infer_perf/general_perf/datasets/open_squad/

# copy open_albert data
mkdir -p ./ByteMLPerf/byte_infer_perf/general_perf/model_zoo/popular/open_albert
cp ./general_perf/model_zoo/popular/open_albert/*.pt ./ByteMLPerf/byte_infer_perf/general_perf/model_zoo/popular/open_albert

# run acc script
cd ./ByteMLPerf/byte_infer_perf/general_perf
mkdir -p workloads
wget -O workloads/albert-torch-fp32.json https://raw.githubusercontent.com/bytedance/ByteMLPerf/refs/heads/main/byte_infer_perf/general_perf/workloads/albert-torch-fp32.json
sed -i 's/tensorrt_legacy/tensorrt/' ./backends/ILUVATAR/common.py
sed -i 's/tensorrt_legacy/tensorrt/' ./backends/ILUVATAR/compile_backend_iluvatar.py
sed -i 's/tensorrt_legacy/tensorrt/' ./backends/ILUVATAR/runtime_backend_iluvatar.py
python3 core/perf_engine.py --hardware_type ILUVATAR --task albert-torch-fp32
```

## Model Results

| Model  | BatchSize | Precision | QPS   | Exact Match | F1 Score |
| ------ | --------- | --------- | ----- | ----------- | -------- |
| ALBERT | 1         | FP16      | 50.99 | 80.18       | 87.57    |
