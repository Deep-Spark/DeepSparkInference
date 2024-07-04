# DeBerta

## Description

DeBERTa (Decoding-enhanced BERT with disentangled attention) is an enhanced version of the BERT (Bidirectional Encoder Representations from Transformers) model. It improves text representation learning by introducing disentangled attention mechanisms and decoding enhancement techniques.DeBERTa introduces disentangled attention mechanisms that decompose the self-attention matrix into different parts, focusing on different semantic information. This helps the model better capture relationships between texts.By incorporating decoding enhancement techniques, DeBERTa adjusts the decoder during fine-tuning to better suit specific downstream tasks, thereby improving the model’s performance on those tasks.

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

Pretrained model: <https://lf-bytemlperf.17mh.cn/obj/bytemlperf-zoo/open_deberta.tar >

Dataset: <https://lf-bytemlperf.17mh.cn/obj/bytemlperf-zoo/open_squad.tar > to download the squad dataset.

or you can :
```bash
bash /scripts/prepare_model_and_dataset.sh

```

### Model Conversion
Please correct the paths in the following commands or files.
```bash
tar -xvf open_deberta.tar
wget < https://github.com/bytedance/ByteMLPerf/blob/main/byte_infer_perf/general_perf/model_zoo/deberta-torch-fp32.json >
python3 torch2onnx.py --model_path deberta-base-squad.pt --output_path deberta-torch-fp32.onnx
onnxsim deberta-torch-fp32.onnx deberta-torch-fp32-sim.onnx
python3 remove_clip_and_cast.py

```

## Inference


```bash
export ORIGIN_ONNX_NAME=/Path/deberta-sim-drop-clip-drop-invaild-cast
export OPTIMIER_FILE=/Path/ixrt/oss/tools/optimizer/optimizer.py
export PROJ_PATH=./
```

### Performance

```bash

bash scripts/infer_deberta_fp16_performance.sh
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
get /upload/3-app/byteperf/Palak.tar
exit
tar -zxvf Palak.tar

接着修改代码：ByteMLPerf/byte_infer_perf/general_perf/datasets/open_squad/data_loader.py
AutoTokenizer.from_pretrained("Palak/microsoft_deberta-base_squad") => AutoTokenizer.from_pretrained("/Your/Path/Palak/microsoft_deberta-base_squad")

mv deberta-sim-drop-clip-drop-invaild-cast.onnx general_perf/model_zoo/popular/open_deberta/
cd /ByteMLPerf/byte_infer_perf/
mv /general_perf/general_perf/model_zoo/popular/open_deberta /general_perf/model_zoo/popular/open_deberta
cd /ByteMLPerf/byte_infer_perf/general_perf
python3 core/perf_engine.py --hardware_type ILUVATAR --task deberta-torch-fp32
```

If report ModuleNotFoundError: No module named 'tensorrt_legacy',Please fix /home/xinchi.tian/ByteMLPerf/byte_infer_perf/general_perf/backends/ILUVATAR/common.py "tensorrt_legacy" to "tensorrt"


## Results

Model   |BatchSize  |Precision |QPS       |Exact Match  |F1 Score
--------|-----------|----------|----------|-------------|------------
DeBerta |    16     |   FP16   | 18.58    | 73.76       | 81.24