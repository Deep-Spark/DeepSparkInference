# Asr transformer fp16 inference （BeamSearch）

## Description

Beam search allows us to exert control over the output of text generation. This is useful because we sometimes know exactly what we want inside the output. For example, in a Neural Machine Translation task, we might know which words must be included in the final translation with a dictionary lookup.


## Setup

### Install

```
pip3 install speechbrain==0.5.13
```

* ixrt 4.0.1_MR release

### Download

Pretrained model: <https://drive.google.com/drive/folders/1_2zN6lbu4zUc0-iq8XbABEm6fl9mohkv>

Dataset: <https://www.openslr.org/resources/33/data_aishell.tgz> to download the Aishell dataset.

```
# Make sure the checkpoint path is results/transformer/8886/save
mkdir -p results/transformer/8886/save
# Make sure the dataset path is results/transformer/8886/save
mkdir -p /home/data/speechbrain
```

## Inference

### Build faster kernels

```bash
bash build.sh
```

### Build engine

max_batch_size and max_seq_len depend on the situation.

```
python3 builder.py \
--ckpt_path results/transformer/8886/save \
--head_num 4 \
--max_batch_size 64  \
--max_seq_len 1024 \
--engine_path transformer.engine
```

### Run engine

```
python3 inference.py hparams/train_ASR_transformer.yaml --data_folder=/home/data/speechbrain/aishell --engine_path transformer.engine 
```