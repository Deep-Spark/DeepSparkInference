# Qwen1.5-32B-Chat (vLLM)

## Model Description

Qwen1.5 is a language model series including decoder language models of different model sizes. For each size, we release
the base language model and the aligned chat model. It is based on the Transformer architecture with SwiGLU activation,
attention QKV bias, group query attention, mixture of sliding window attention and full attention, etc. Additionally, we
have an improved tokenizer adaptive to multiple natural languages and codes. 

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.09 |
| MR-V100 | 4.2.0 | 25.03 |

## Model Preparation

### Prepare Resources

- Model: <https://modelscope.cn/models/Qwen/Qwen1.5-32B-Chat>

```bash
cd ${DeepSparkInference}/models/nlp/large_language_model/qwen1.5-32b/vllm
mkdir -p data/qwen1.5
ln -s /path/to/Qwen1.5-32B ./data/qwen1.5
```

### Install Dependencies

In order to run the model smoothly, you need to get the sdk from [resource
center](https://support.iluvatar.com/#/ProductLine?id=2) of Iluvatar CoreX official website.

## Model Inference

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 offline_inference.py --model ./data/qwen1.5/Qwen1.5-32B-Chat --max-tokens 256 -tp 4 --temperature 0.0
```
