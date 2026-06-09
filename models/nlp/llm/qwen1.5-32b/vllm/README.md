# Qwen1.5-32B-Chat (vLLM)

## Model Description

Qwen1.5 is a language model series including decoder language models of different model sizes. For each size, we release
the base language model and the aligned chat model. It is based on the Transformer architecture with SwiGLU activation,
attention QKV bias, group query attention, mixture of sliding window attention and full attention, etc. Additionally, we
have an improved tokenizer adaptive to multiple natural languages and codes. 

## Supported Environments

| GPU | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release | Branch |
| :----: | :----: | :----: | :----: |
| MR-V100 | 4.4.0 | 26.03 | release/26.03 |
| MR-V100 | 4.3.0 | 25.12 | release/25.12 |

> **Note:** 请切换到与您的 SDK 版本对应的 Release 分支进行测试。请勿直接在 master 分支上运行测试，因为 master 分支可能包含与您的本地 SDK 版本不兼容的最新更改。
>
> 切换分支命令示例：`git checkout release/26.03`

## Model Preparation

### Prepare Resources

- Model: <https://modelscope.cn/models/Qwen/Qwen1.5-32B-Chat>

```bash
cd ${DeepSparkInference}/models/nlp/llm/qwen1.5-32b/vllm
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

### Benchmarking Results

| Model | Precision  | QPS | Total TPS | Output TPS |
| :----: | :----: | :----: | :----: | :----: |
| Qwen1.5-32B | BF16 | 1.1 | 756.57 | 164.26 |