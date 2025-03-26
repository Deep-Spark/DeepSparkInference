# Transformer ASR (IxRT)

## Model Description

Beam search allows us to exert control over the output of text generation. This is useful because we sometimes know
exactly what we want inside the output. For example, in a Neural Machine Translation task, we might know which words
must be included in the final translation with a dictionary lookup.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| MR-V100 | 4.2.0     |  25.03  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://drive.google.com/drive/folders/1_2zN6lbu4zUc0-iq8XbABEm6fl9mohkv>

Dataset: <https://www.openslr.org/resources/33/data_aishell.tgz> to download the Aishell dataset.

```bash
# Make sure the checkpoint path is results/transformer/8886/save
mkdir -p results/transformer/8886/save
# The data path like below:
results/transformer/8886
├── cer.txt
├── dev.csv
├── env.log
├── hyperparams.yaml
├── inference_encoder_ctc.py
├── inference.py
├── log.txt
├── save
│   ├── CKPT+2023-03-29+06-31-40+00
│   │   ├── brain.ckpt
│   │   ├── CKPT.yaml
│   │   ├── counter.ckpt
│   │   ├── model.ckpt
│   │   ├── noam_scheduler.ckpt
│   │   └── normalizer.ckpt
│   └── tokenizer.ckpt
├── test.csv
├── train.csv
└── train_log.txt

# Make sure the dataset path is results/transformer/8886/save
mkdir -p /home/data/speechbrain/aishell/csv_data
ln -s /PATH/to/data_aishell /home/data/speechbrain/aishell/
cp results/transformer/8886/*.csv /home/data/speechbrain/aishell/csv_data
```

### Install Dependencies

```bash
pip3 install -r requirements.txt
```

## Model Inference

### Build faster kernels

```bash
bash build.sh
```

### Build engine

max_batch_size and max_seq_len depend on the situation.

```bash
python3 builder.py \
--ckpt_path results/transformer/8886/save \
--head_num 4 \
--max_batch_size 64  \
--max_seq_len 1024 \
--engine_path transformer.engine
```

### Run engine

```bash
python3 inference.py hparams/train_ASR_transformer.yaml --data_folder=/home/data/speechbrain/aishell --engine_path transformer.engine 
```

## Model Results

| Model           | BatchSize | Precision | QPS   | CER  |
|-----------------|-----------|-----------|-------|------|
| Transformer ASR | 32        | FP16      | 15.64 | 5.95 |
