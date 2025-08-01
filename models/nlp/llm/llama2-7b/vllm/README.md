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
cd ${DeepSparkInference}/models/nlp/large_language_model/llama2-7b/vllm
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
