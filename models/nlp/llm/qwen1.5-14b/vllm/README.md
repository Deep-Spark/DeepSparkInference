# Qwen1.5-14B (vLLM)

## Model Description

Qwen1.5 is a language model series including decoder language models of different model sizes. For each size, we release
the base language model and the aligned chat model. It is based on the Transformer architecture with SwiGLU activation,
attention QKV bias, group query attention, mixture of sliding window attention and full attention, etc. Additionally, we
have an improved tokenizer adaptive to multiple natural languages and codes. For the beta version, temporarily we did
not include GQA (except for 32B) and the mixture of SWA and full attention.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| MR-V100 | 4.2.0     |  25.03  |

## Model Preparation

### Prepare Resources

- Model: <https://modelscope.cn/models/qwen/Qwen1.5-14B/summary>

```bash
cd ${DeepSparkInference}/models/nlp/large_language_model/qwen1.5-14b/vllm
mkdir data/qwen1.5
ln -s /path/to/Qwen1.5-14B ./data/qwen1.5
```

### Install Dependencies

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx
```

## Model Inference

```bash
python3 offline_inference.py --model ./data/qwen1.5/Qwen1.5-14B --max-tokens 256 -tp 1 --temperature 0.0 --max-model-len 896
```

## Model Results

| Model       | QPS   |
|-------------|-------|
| Qwen1.5-14B | 57.15 |
