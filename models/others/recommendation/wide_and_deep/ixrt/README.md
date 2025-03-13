# Wide & Deep (IxRT)

## Model Description

Generalized linear models with nonlinear feature transformations are widely used for large-scale regression and classification problems with sparse inputs. Memorization of feature interactions through a wide set of cross-product feature transformations are effective and interpretable, while generalization requires more feature engineering effort. With less feature engineering, deep neural networks can generalize better to unseen feature combinations through low-dimensional dense embeddings learned for the sparse features. However, deep neural networks with embeddings can over-generalize and recommend less relevant items when the user-item interactions are sparse and high-rank. In this paper, we present Wide & Deep learning---jointly trained wide linear models and deep neural networks---to combine the benefits of memorization and generalization for recommender systems. We productionized and evaluated the system on Google Play, a commercial mobile app store with over one billion active users and over one million apps. Online experiment results show that Wide & Deep significantly increased app acquisitions compared with wide-only and deep-only models. We have also open-sourced our implementation in TensorFlow.

## Supported Environments

| Iluvatar GPU | IXUCA SDK |
|--------------|-----------|
| MR-V100      | 4.2.0     |

## Model Preparation

### Prepare Resources

Pretrained model: <https://lf-bytemlperf.17mh.cn/obj/bytemlperf-zoo/open_wide_deep_saved_model.tar>

Dataset: <https://lf-bytemlperf.17mh.cn/obj/bytemlperf-zoo/eval.csv>

### Install Dependencies

```bash
apt install -y libnuma-dev

pip3 install -r requirements.txt
```

### Model Conversion

```bash
# Go to path of this model
export PROJ_ROOT=/PATH/TO/DEEPSPARKINFERENCE
export MODEL_PATH=${PROJ_ROOT}/models/recommendation/widedeep/ixrt
cd ${MODEL_PATH}

# export onnx
python3 export_onnx.py --model_path open_wide_deep_saved_model --output_path open_wide_deep_saved_model/widedeep.onnx

# Simplify onnx model
onnxsim open_wide_deep_saved_model/widedeep.onnx open_wide_deep_saved_model/widedeep_sim.onnx
python3 deploy.py --model_path open_wide_deep_saved_model/widedeep_sim.onnx --output_path open_wide_deep_saved_model/widedeep_sim.onnx
python3 change2dynamic.py --model_path open_wide_deep_saved_model/widedeep_sim.onnx --output_path open_wide_deep_saved_model/widedeep_sim.onnx
```

## Model Inference

```bash
export ORIGIN_ONNX_NAME=./open_wide_deep_saved_model/widedeep_sim
export OPTIMIER_FILE=${IXRT_OSS_ROOT}/tools/optimizer/optimizer.py
export PROJ_PATH=./
```

### Performance

#### FP16

```bash
bash scripts/infer_widedeep_fp16_performance.sh
```

### Accuracy

If you want to evaluate the accuracy of this model, please visit the website: <https://github.com/yudefu/ByteMLPerf/tree/iluvatar_general_infer>, which integrates inference and training of many models under this framework, supporting the ILUVATAR backend

For detailed steps regarding this model, please refer to this document: <https://github.com/yudefu/ByteMLPerf/blob/iluvatar_general_infer/byte_infer_perf/general_perf/backends/ILUVATAR/README.zh_CN.md> Note: You need to modify the relevant paths in the code to your own correct paths.

```bash
# link and install ByteMLPerf requirements
ln -s ${PROJ_ROOT}/toolbox/ByteMLPerf ./
pip3 install -r ./ByteMLPerf/byte_infer_perf/general_perf/requirements.txt

# Get eval.csv and onnx
mkdir -p ./ByteMLPerf/byte_infer_perf/general_perf/model_zoo/regular/open_wide_deep_saved_model
mkdir -p ./ByteMLPerf/byte_infer_perf/general_perf/datasets/open_criteo_kaggle/

wget https://lf-bytemlperf.17mh.cn/obj/bytemlperf-zoo/eval.csv
mv eval.csv ./ByteMLPerf/byte_infer_perf/general_perf/datasets/open_criteo_kaggle/

wget http://files.deepspark.org.cn:880/deepspark/widedeep_dynamicshape_new.onnx
cp open_wide_deep_saved_model/* ./ByteMLPerf/byte_infer_perf/general_perf/model_zoo/regular/open_wide_deep_saved_model/
mv widedeep_dynamicshape_new.onnx ./ByteMLPerf/byte_infer_perf/general_perf/model_zoo/regular/open_wide_deep_saved_model/widedeep_dynamicshape.onnx

# Run Acc scripts
cd ./ByteMLPerf/byte_infer_perf/general_perf
mkdir -p workloads
wget -O workloads/widedeep-tf-fp32.json https://raw.githubusercontent.com/bytedance/ByteMLPerf/refs/heads/main/byte_infer_perf/general_perf/workloads/widedeep-tf-fp32.json
python3 core/perf_engine.py --hardware_type ILUVATAR --task widedeep-tf-fp32
```

## Model Results

| Model       | BatchSize | Precision | FPS      | ACC     |
|-------------|-----------|-----------|----------|---------|
| Wide & Deep | 1024      | FP16      | 77073.93 | 0.74597 |
