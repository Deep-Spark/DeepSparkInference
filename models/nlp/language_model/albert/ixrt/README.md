# AlBERT

## Description

Albert (A Lite BERT) is a variant of the BERT (Bidirectional Encoder Representations from Transformers) model that focuses on efficiency and scalability while maintaining strong performance in natural language processing tasks. The AlBERT model introduces parameter reduction techniques and incorporates self-training strategies to enhance its effectiveness.

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

Pretrained model: <https://lf-bytemlperf.17mh.cn/obj/bytemlperf-zoo/open_albert.tar>

Dataset: <https://lf-bytemlperf.17mh.cn/obj/bytemlperf-zoo/open_squad.tar > to download the squad dataset.

or you can :
```bash
bash /scripts/prepare_model_and_dataset.sh

```

### Model Conversion
Please correct the paths in the following commands or files.
```bash
tar -xvf open_albert.tar
wget < https://github.com/bytedance/ByteMLPerf/blob/main/byte_infer_perf/general_perf/model_zoo/albert-torch-fp32.json >
python3 torch2onnx.py --model_path albert-base-squad.pt --output_path albert-torch-fp32.onnx
onnxsim albert-torch-fp32.onnx albert-torch-fp32-sim.onnx

```

## Inference


```bash
export ORIGIN_ONNX_NAME=/Path/albert-base-squad
export OPTIMIER_FILE=/Path/ixrt/oss/tools/optimizer/optimizer.py
export PROJ_PATH=./
```

### Performance

```bash

bash scripts/infer_albert_fp16_performance.sh
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
sftp -P 29880 vipzjtd@iftp.iluvatar.com.cn     密码：123..com
get /upload/3-app/byteperf/madlag.tar
exit
tar -zxvf madlag.tar

接着修改代码：ByteMLPerf/byte_infer_perf/general_perf/datasets/open_squad/data_loader.py
AutoTokenizer.from_pretrained("madlag/albert-base-v2-squad") => AutoTokenizer.from_pretrained("/Your/Path/madlag/albert-base-v2-squad")

cd /ByteMLPerf/byte_infer_perf/
mv /general_perf/general_perf/model_zoo/popular/open_albert /general_perf/model_zoo/popular/open_albert
cd /ByteMLPerf/byte_infer_perf/general_perf
python3 core/perf_engine.py --hardware_type ILUVATAR --task albert-torch-fp32
```

If report ModuleNotFoundError: No module named 'tensorrt_legacy',Please fix:
<ByteMLPerf/byte_infer_perf/general_perf/backends/ILUVATAR/common.py> "tensorrt_legacy" to "tensorrt"
<ByteMLPerf/byte_infer_perf/general_perf/backends/ILUVATAR/compile_backend_iluvatar.py> "tensorrt_legacy" to "tensorrt"
<ByteMLPerf/byte_infer_perf/general_perf/backends/ILUVATAR/runtime_backend_iluvatar.py> "tensorrt_legacy" to "tensorrt"


## Results

Model   |BatchSize  |Precision |QPS       |Exact Match  |F1 Score
--------|-----------|----------|----------|-------------|------------
AlBERT  |    16     |   FP16   | 50.99    | 80.18       | 87.57