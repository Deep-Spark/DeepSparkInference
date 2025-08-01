# Qwen2-72B-Instruct (vLLM)

## Model Description

Qwen2 is the new series of Qwen large language models. For Qwen2, we release a number of base language models and
instruction-tuned language models ranging from 0.5 to 72 billion parameters, including a Mixture-of-Experts model. This
repo contains the instruction-tuned 72B Qwen2 model.

Compared with the state-of-the-art opensource language models, including the previous released Qwen1.5, Qwen2 has
generally surpassed most opensource models and demonstrated competitiveness against proprietary models across a series
of benchmarks targeting for language understanding, language generation, multilingual capability, coding, mathematics,
reasoning, etc.

Qwen2-72B-Instruct supports a context length of up to 131,072 tokens, enabling the processing of extensive inputs.
Please refer to this section for detailed instructions on how to deploy Qwen2 for handling long texts.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.09 |
| MR-V100 | 4.2.0 | 25.03 |

## Model Preparation

### Prepare Resources

- Model: <https://modelscope.cn/models/Qwen/Qwen2-72B-Instruct>

```bash
cd ${DeepSparkInference}/models/nlp/large_language_model/qwen2-72b/vllm
mkdir -p data/qwen2
ln -s /path/to/Qwen2-72B ./data/qwen2
```

### Install Dependencies

In order to run the model smoothly, you need to get the sdk from [resource
center](https://support.iluvatar.com/#/ProductLine?id=2) of Iluvatar CoreX official website.

## Model Inference

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 offline_inference.py --model ./data/qwen2/Qwen2-72B --max-tokens 256 -tp 8 --temperature 0.0 --gpu-memory-utilization 0.98 --max-model-len 32768
```
