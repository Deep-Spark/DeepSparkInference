# Conformer

## Description

Conformer is a speech recognition model proposed by Google in 2020. It combines the advantages of CNN and Transformer. CNN efficiently extracts local features, while Transformer is more effective in capturing long sequence dependencies. Conformer applies convolution to the Encoder layer of Transformer, enhancing the performance of Transformer in the ASR (Automatic Speech Recognition) domain.

## Setup

### Install

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

pip3 install -r requirements.txt
```

### Download

Pretrained model: <https://github.com/wenet-e2e/wenet/blob/main/docs/pretrained_models.en.md>

Dataset: <https://www.openslr.org/33/> to download the Aishell dataset.

Download and put model in conformer_checkpoints.

```bash
ln -s /home/deepspark/datasets/INFER/conformer/20210601_u2++_conformer_exp_aishell ./conformer_checkpoints
```

### Prepare Data

```bash
# Accuracy
DATA_DIR=/PATH/to/aishell_test_data
TOOL_DIR="$(pwd)/tools"
bash scripts/aishell_data_prepare.sh ${DATA_DIR} ${TOOL_DIR}
```

## Model Conversion And Inference

### FP16

```bash
# Accuracy
bash scripts/infer_conformer_fp16_accuracy.sh
# Performance
bash scripts/infer_conformer_fp16_performance.sh
```

## Results

| Model     | BatchSize | Precision | QPS     | CER    |
| --------- | --------- | --------- | ------- | ------ |
| Conformer | 24        | FP16      | 387.821 | 0.0517 |
