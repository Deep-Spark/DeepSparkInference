# DeepSpeech2 (ixRT)

## Model Description

DeepSpeech2 is an end-to-end speech recognition model based on RNNs and CTC decoding, developed by Baidu. It uses CNN for acoustic feature extraction followed by RNN encoders and CTC decoder.

## Supported Environments

| GPU | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release | Branch |
| :----: | :----: | :----: | :----: |
| MR-V100 | 4.4.0 | 26.06 | release/26.06 |

> **Note:** 请切换到与您的 SDK 版本对应的 Release 分支进行测试。请勿直接在 master 分支上运行测试，因为 master 分支可能包含与您的本地 SDK 版本不兼容的最新更改。
>
> 切换分支命令示例：`git checkout release/26.06`

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

echo "Build engine!"
python3 modify_model_to_dynamic.py --static_onnx checkpoints/deepspeech2.onnx --dynamic_onnx checkpoints/deepspeech2_dynamic.onnx
python3 build_engine.py \
    --model_name deepspeech2 \
    --onnx_path checkpoints/deepspeech2_dynamic.onnx \
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
