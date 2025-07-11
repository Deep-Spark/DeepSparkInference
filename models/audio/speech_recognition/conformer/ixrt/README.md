# Conformer (IxRT)

## Model Description

Conformer is a speech recognition model proposed by Google in 2020. It combines the advantages of CNN and Transformer. CNN efficiently extracts local features, while Transformer is more effective in capturing long sequence dependencies. Conformer applies convolution to the Encoder layer of Transformer, enhancing the performance of Transformer in the ASR (Automatic Speech Recognition) domain.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.09 |
| MR-V100 | 4.2.0 | 25.03 |

## Model Preparation

### Prepare Resources

Pretrained model: <https://github.com/wenet-e2e/wenet/blob/main/docs/pretrained_models.en.md>

Dataset: <https://www.openslr.org/33/> to download the Aishell dataset.

```bash
# Download and put model in conformer_checkpoints
wget http://files.deepspark.org.cn:880/deepspark/conformer_checkpoints.tar.gz
tar xf conformer_checkpoints.tar.gz

# Prepare AISHELL Data
DATA_DIR=/PATH/to/aishell_test_data
TOOL_DIR="$(pwd)/tools"
bash scripts/aishell_data_prepare.sh ${DATA_DIR} ${TOOL_DIR}
```

### Install Dependencies

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

pip3 install -r requirements.txt
```

## Model Inference

### FP16

```bash
# Accuracy
bash scripts/infer_conformer_fp16_accuracy.sh
# Performance
bash scripts/infer_conformer_fp16_performance.sh
```

## Model Results

| Model     | BatchSize | Precision | QPS     | CER    |
| --------- | --------- | --------- | ------- | ------ |
| Conformer | 24        | FP16      | 387.821 | 0.0517 |
