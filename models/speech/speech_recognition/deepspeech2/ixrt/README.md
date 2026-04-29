# DeepSpeech2 (ixRT)

## Model Description

DeepSpeech2 is an end-to-end speech recognition model based on RNNs and CTC decoding, developed by Baidu. It uses CNN for acoustic feature extraction followed by RNN encoders and CTC decoder.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.4.0 | 26.06 |

## Model Preparation

### Prepare Resources

Pretrained model: <http://files.deepspark.org.cn:880/deepspark/data/checkpoints/deepspeech2.onnx>

Dataset: LibriSpeech <http://files.deepspark.org.cn:880/deepspark/data/datasets/LibriSpeech.tar.gz>

### Install Dependencies

Contact the Iluvatar administrator to get the missing packages:
- paddlepaddle-*.whl

```bash
pip3 install librosa psutil pysoundfile pytest requests tensorboardX editdistance textgrid onnxsim paddlespeech_ctcdecoders paddleaudio paddlespeech
pip3 install numpy==1.23.5
```

### Model Conversion

```bash
mkdir checkpoints
cd checkpoints
wget http://files.deepspark.org.cn:880/deepspark/data/checkpoints/deepspeech2.onnx
wget http://files.deepspark.org.cn:880/deepspark/data/checkpoints/common_crawl_00.prune01111.trie.klm


git clone https://gitee.com/deep-spark/iluvatar-corex-ixrt.git --depth=1

OPTIMIER_FILE=iluvatar-corex-ixrt/tools/optimizer/optimizer.py
echo "Build engine!"
python3 modify_model_to_dynamic.py --static_onnx checkpoints/deepspeech2.onnx --dynamic_onnx checkpoints/deepspeech2_dynamic.onnx
python3 ${OPTIMIER_FILE}  --onnx checkpoints/deepspeech2_dynamic.onnx --model_type rnn --not_sim
python3 build_engine.py \
    --model_name deepspeech2 \
    --onnx_path checkpoints/deepspeech2_dynamic_end.onnx \
    --engine_path checkpoints/deepspeech2.engine

```

## Model Inference

```bash
export DATASETS_DIR=/path/to/LibriSpeech/
export CHECKPOINTS_DIR=./checkpoints
export RUN_DIR=./
```

### FP16

```bash
# Test ACC (WER)
bash scripts/infer_deepspeech2_fp16_accuracy.sh
# Test FPS
bash scripts/infer_deepspeech2_fp16_performance.sh
```

## Model Results

| Model        | BatchSize | Precision | ThroughPut    | WER(%) |
| ------------ | --------- | --------- | ------- | ------ |
| DeepSpeech2  | 1         | FP16      | 1584.153  | 5.8    |
