# Wide & Deep

## Description

Generalized linear models with nonlinear feature transformations are widely used for large-scale regression and classification problems with sparse inputs. Memorization of feature interactions through a wide set of cross-product feature transformations are effective and interpretable, while generalization requires more feature engineering effort. With less feature engineering, deep neural networks can generalize better to unseen feature combinations through low-dimensional dense embeddings learned for the sparse features. However, deep neural networks with embeddings can over-generalize and recommend less relevant items when the user-item interactions are sparse and high-rank. In this paper, we present Wide & Deep learning---jointly trained wide linear models and deep neural networks---to combine the benefits of memorization and generalization for recommender systems. We productionized and evaluated the system on Google Play, a commercial mobile app store with over one billion active users and over one million apps. Online experiment results show that Wide & Deep significantly increased app acquisitions compared with wide-only and deep-only models. We have also open-sourced our implementation in TensorFlow.

## Setup

### Install

```bash
pip3 install tf2onnx
pip3 install onnxsim
pip3 install numa

```

### Download

Pretrained model: <https://lf-bytemlperf.17mh.cn/obj/bytemlperf-zoo/open_wide_deep_saved_model.tar>

Dataset: <https://lf-bytemlperf.17mh.cn/obj/bytemlperf-zoo/eval.csv >

```bash

# export onnx
python3 export_onnx.py --model_path open_wide_deep_saved_model --output_path open_wide_deep_saved_model/widedeep.onnx

# Simplify onnx model
onnxsim open_wide_deep_saved_model/widedeep.onnx open_wide_deep_saved_model/widedeep_sim.onnx
python3 deploy.py --model_path open_wide_deep_saved_model/widedeep_sim.onnx --output_path open_wide_deep_saved_model/widedeep_sim.onnx
python3 change2dynamic.py --model_path open_wide_deep_saved_model/widedeep_sim.onnx --output_path open_wide_deep_saved_model/widedeep_sim.onnx
```

## Inference

```bash
export ORIGIN_ONNX_NAME=/Path/widedeep_sim
export OPTIMIER_FILE=/Path/ixrt/oss/tools/optimizer/optimizer.py
export PROJ_PATH=./
```

### Performance

```bash
bash scripts/infer_widedeep_fp16_performance.sh
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

mkdir -p ./ByteMLPerf/byte_infer_perf/general_perf/model_zoo/regular/open_wide_deep_saved_model
mkdir -p ./ByteMLPerf/byte_infer_perf/general_perf/datasets/open_criteo_kaggle/
wget -O ./ByteMLPerf/byte_infer_perf/general_perf/datasets/open_criteo_kaggle/eval.csv https://lf-bytemlperf.17mh.cn/obj/bytemlperf-zoo/eval.csv

sftp -P 29889 user01@58.247.142.52  password：5$gS%659
cd yudefu/bytedance_perf ; get widedeep_dynamicshape_new.onnx
exit

mv path/to/widedeep_dynamicshape_new.onnx ./ByteMLPerf/byte_infer_perf/general_perf/model_zoo/regular/open_wide_deep_saved_model/widedeep_dynamicshape.onnx
cd ./ByteMLPerf/byte_infer_perf/general_perf
python3 core/perf_engine.py --hardware_type ILUVATAR --task widedeep-tf-fp32
```