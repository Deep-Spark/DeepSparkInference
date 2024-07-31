# Conformer

## Description

Conformer is a speech recognition model proposed by Google in 2020. It combines the advantages of CNN and Transformer. CNN efficiently extracts local features, while Transformer is more effective in capturing long sequence dependencies. Conformer applies convolution to the Encoder layer of Transformer, enhancing the performance of Transformer in the ASR (Automatic Speech Recognition) domain.

## Setup

### Install

```bash
pip3 install tqdm
pip3 install onnx
pip3 install typeguard==2.13.3
pip3 install onnxsim
```

### Download

Pretrained model: <https://github.com/wenet-e2e/wenet/blob/main/docs/pretrained_models.md>

Dataset: <https://www.openslr.org/33/> to download the Aishell dataset.

download and put model in conformer_checkpoints, put data in aishell_test_data.

### Prepare Data
```bash
# Accuracy
DATA_DIR=./aishell_test_data
Tool_DIR=./tools
bash scripts/aishell_data_prepare.sh ${DATA_DIR} ${Tool_DIR}
```

### Model Conversion And Inference

### FP16

```bash
# Accuracy
bash scripts/infer_conformer_fp16_accuracy_ixrt.sh
# Performance
bash scripts/infer_conformer_fp16_performance_ixrt.sh
```

## Results

Model      |BatchSize  |Precision |QPS       |CER       |
-----------|-----------|----------|----------|----------|
Conformer  |    24     |   FP16   | 380.00 |  0.051   |
