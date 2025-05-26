# Baichuan2-7B (vLLM)

## Model Description

Baichuan 2 is a new generation open-source large language model launched by Baichuan Intelligence. It is trained on
high-quality data with 26 trillion tokens, which sounds like a substantial dataset. Baichuan 2 achieves state-of-the-art
performance on various authoritative Chinese, multilingual, and domain-specific benchmarks of similar size, indicating
its excellent capabilities in language understanding and generation.This release includes Base and Chat versions of 7B.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.2.0     |  25.03  |

## Model Preparation

### Prepare Resources

Pretrained model:
[https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/tree/main](https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/tree/main)

```bash
mkdir /data/baichuan/
mv Baichuan2-7B-Base.tar/zip /data/baichuan/
```

### Install Dependencies

In order to run the model smoothly, you need to get the sdk from [resource
center](https://support.iluvatar.com/#/ProductLine?id=2) of Iluvatar CoreX official website.

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

pip3 install transformers
```

## Model Inference

```bash
python3 offline_inference.py --model /data/baichuan/Baichuan2-7B-Base/ --max-tokens 256 --trust-remote-code --chat_template template_baichuan.jinja --temperature 0.0
```

### Run Baichuan w8a16 quantization

Retrieve int8 weights.

Int8 weights will be saved at /data/baichuan/Baichuan2-7B-Base/int8

```bash
python3 convert2int8.py --model-path /data/baichuan/Baichuan2-7B-Base/
```

```bash
python3 offline_inference.py --model /data/baichuan/Baichuan2-7B-Base/int8/ --chat_template template_baichuan.jinja --quantization w8a16 --max-num-seqs 1 --max-model-len 256 --trust-remote-code --temperature 0.0 --max-tokens 256
```

## Model Results

| Model        | Precision | tokens | QPS    |
| :----: | :----: | :----: | :----: |
| Baichuan2-7B | FP16      | 768    | 109.27 |
| Baichuan2-7B | w8a16     | 740    | 59.82  |
