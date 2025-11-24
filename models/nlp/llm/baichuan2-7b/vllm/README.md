# Baichuan2-7B (vLLM)

## Model Description

Baichuan 2 is a new generation open-source large language model launched by Baichuan Intelligence. It is trained on
high-quality data with 26 trillion tokens, which sounds like a substantial dataset. Baichuan 2 achieves state-of-the-art
performance on various authoritative Chinese, multilingual, and domain-specific benchmarks of similar size, indicating
its excellent capabilities in language understanding and generation.This release includes Base and Chat versions of 7B.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.09 |
| MR-V100 | 4.2.0 | 25.03 |

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

### Benchmarking vLLM

```bash
git clone https://github.com/vllm-project/vllm.git -b v0.8.3 --depth=1
python3 vllm/benchmarks/benchmark_throughput.py \
  --model {model_name} \
  --dataset-name sonnet \
  --dataset-path vllm/benchmarks/sonnet.txt \
  --num-prompts 10
```

If you raise "AttributeError: BaichuanTokenizer has no attribute default_chat_template.", please add below code into tokenizer_config.json

```json
"chat_template": "{{ (messages|selectattr('role', 'equalto', 'system')|list|last).content|trim if (messages|selectattr('role', 'equalto', 'system')|list) else '' }}{%- for message in messages -%}{%- if message['role'] == 'user' -%}{{- '<reserved_106>' + message['content'] -}}{%- elif message['role'] == 'assistant' -%}{{- '<reserved_107>' + message['content'] -}}{%- endif -%}{%- endfor -%}{%- if add_generation_prompt and messages[-1]['role'] != 'assistant' -%}{{- '<reserved_107>' -}}{% endif %}"
```

### Benchmarking Results

| Model | Precision  | QPS | Total TPS | Output TPS |
| :----: | :----: | :----: | :----: | :----: |
| Baichuan2-7B | FP16     | 1.69  | 1168.23    | 252.90 |
| Baichuan2-7B | w8a16    | 0.248  | 740    | 61.318  |
