# Llama2 7B (vLLM)

## Model Description

we develop and release Llama 2, a collection of pretrained and fine-tuned large language models (LLMs) ranging in scale
from 7 billion to 70 billion parameters. Our fine-tuned LLMs, called Llama 2-Chat, are optimized for dialogue use cases.
Our models outperform open-source chat models on most benchmarks we tested, and based on our human evaluations for
helpfulness and safety, may be a suitable substitute for closed-source models. We provide a detailed description of our
approach to fine-tuning and safety improvements of Llama 2-Chat in order to enable the community to build on our work
and contribute to the responsible development of LLMs.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.09 |
| MR-V100 | 4.2.0 | 25.03 |

## Model Preparation

### Prepare Resources

- Model: <https://huggingface.co/meta-llama/Llama-2-7b>

```bash
cd ${DeepSparkInference}/models/nlp/llm/llama2-7b/vllm
mkdir -p data/llama2
ln -s /path/to/llama2-7b ./data/llama2
```

### Install Dependencies

In order to run the model smoothly, you need to get the sdk from [resource
center](https://support.iluvatar.com/#/ProductLine?id=2) of Iluvatar CoreX official website.

## Model Inference

```bash
python3 offline_inference.py --model ./data/llama2/llama2-7b --max-tokens 256 -tp 1 --temperature 0.0
python3 offline_inference.py --model ./data/llama2/llama2-7b --max-tokens 256 -tp 2 --temperature 0.0
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

If you raise "AttributeError: LlamaTokenizerFast has no attribute default_chat_template.", please add below code into tokenizer_config.json

```json
"chat_template": "{{- '<s>[INST] ' -}}{%- for message in messages -%}{%- if loop.first and message['role'] == 'system' -%}{{- '<<SYS>>\n' + message['content'] + '\n<</SYS>>\n' -}}{%- elif message['role'] == 'user' and loop.index <= 2 -%}{{- message['content'] + ' [/INST]' -}}{%- elif message['role'] == 'user' -%}{{- '<s>[INST] ' + message['content'] + ' [/INST]' -}}{%- elif message['role'] == 'assistant' -%}{{- ' ' + message['content'] + ' </s>' -}}{%- endif -%}{%- endfor -%}",
```

### Benchmarking Results

| Model | Precision  | QPS | Total TPS | Output TPS |
| :----: | :----: | :----: | :----: | :----: |
| llama2-7b | FP16     | 1.76  | 1171.87    | 264.29 |