# Qwen3-32B (vLLM)

## Model Description

Qwen3-32B is a dense large language model with 32B parameters, offering excellent performance on reasoning, instruction-following, and multilingual tasks. It supports seamless switching between thinking mode (for complex logical reasoning, math, and coding) and non-thinking mode (for efficient, general-purpose dialogue).

This version supports W8A8 (Weight-8bit, Activation-8bit) and W4A16 (Weight-4bit, Activation-16bit) quantization for efficient inference.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.4.0 | 26.03 |

## Model Preparation

### Prepare Resources

- Model: <https://www.modelscope.cn/models/Qwen/Qwen3-32B>

### Install Dependencies

In order to run the model smoothly, you need to get the sdk from [resource center](https://support.iluvatar.com/#/ProductLine?id=2) of Iluvatar CoreX official website.

## Model Inference

### Inference with W8A8/W4A16

#### Performance Test

1. Use the pre-copied ``llm-benchmark``:
```bash
cd ../../llm-benchmark
pip3 install -r requirements.txt
```

2. Set environment variables:
```bash
export VLLM_ENFORCE_CUDA_GRAPH=1
```

4. Start server:
```bash
vllm serve /path/to/model --trust-remote-code --pipeline-parallel-size=1 --tensor-parallel-size=2 --max-model-len 8192 --gpu-memory-utilization 0.9 --disable-cascade-attn --no-enable-prefix-caching --no-enable_chunked_prefill --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "level": 0}'
```

5. Run client (Input2048, Output1024, BS1):
```bash
./iluvatar_bench sgl-perf --backend vllm --host 0.0.0.0 --port 8000 --model /path/to/model --dataset-name random --dataset-path /path/to/ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 1 --random-input 2048 --max-concurrency 1 --tokenize-prompt --random-range-ratio 1 --random-output 1024
```

## References

- [Qwen3](https://github.com/QwenLM/Qwen3)
- [vLLM](https://github.com/vllm-project/vllm)