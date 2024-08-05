# VideoBERT

## Description

VideoBERT is a model designed for video understanding tasks, extending the capabilities of BERT (Bidirectional Encoder Representations from Transformers) to video data. It enhances video representation learning by integrating both visual and textual information into a unified framework.

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

Pretrained model: <https://lf-bytemlperf.17mh.cn/obj/bytemlperf-zoo/open_videobert.tar>

Dataset: <<https://lf-bytemlperf.17mh.cn/obj/bytemlperf-zoo/cifar-100-python.tar>  > to download the cifar-100-python dataset.

or you can :

```bash
bash /scripts/prepare_model_and_dataset.sh

```

## Inference

```bash
export ORIGIN_ONNX_NAME=/Path/video-bert
export OPTIMIER_FILE=/Path/ixrt/oss/tools/optimizer/optimizer.py
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

pip3 install -r toolbox/ByteMLPerf/byte_infer_perf/general_perf/requirements.txt
mv /ixrt/perf_engine.py toolbox/ByteMLPerf/byte_infer_perf/general_perf/core/perf_engine.py
cd /toolbox/ByteMLPerf/byte_infer_perf/
mv /general_perf/general_perf/model_zoo/popular/open_videobert /general_perf/model_zoo/popular/open_videobert
cd /toolbox/ByteMLPerf/byte_infer_perf/general_perf
python3 core/perf_engine.py --hardware_type ILUVATAR --task videobert-onnx-fp32
```

Modify the <model> variable in the <optimize_to_ixrt> function of the <toolbox/ByteMLPerf/byte_infer_perf/general_perf/backends/ILUVATAR/optimizer/optimizer.py> file to the actual video-bert.onnx path.

## Results

| Model     | BatchSize | Precision | QPS   | Top-1 ACC |
| --------- | --------- | --------- | ----- | --------- |
| VideoBERT | 16        | FP16      | 37.68 | 61.67     |
