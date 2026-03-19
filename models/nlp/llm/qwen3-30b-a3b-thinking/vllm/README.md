# Qwen3-30B-A3B-Thinking-2507 (vLLM)

## Model Description

Qwen3-30B-A3B is a Mixture-of-Experts (MoE) large language model with 30B total parameters and 3B activated parameters. The "Thinking" version is optimized for complex logical reasoning, math, and coding tasks.

This version supports W4A8 (Weight-4bit, Activation-8bit) quantization for efficient inference.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.4.0 | 26.03 |

## Model Preparation

### Prepare Resources

- Model: <https://www.modelscope.cn/models/Qwen/Qwen3-30B-A3B>

### Install Dependencies

In order to run the model smoothly, you need to get the sdk from [resource center](https://support.iluvatar.com/#/ProductLine?id=2) of Iluvatar CoreX official website.

## Model Inference

### Inference with W4A8

#### Performance Test

1. Use the pre-copied ``llm-benchmark``:
```bash
cd ../../llm-benchmark
pip3 install -r requirements.txt
```

2. Set environment variables:
```bash
export VLLM_ENFORCE_CUDA_GRAPH=1
export VLLM_W8A8_MOE_USE_W4A8=1
```

4. Start server:
```bash
vllm serve /path/to/model --trust-remote-code --pipeline-parallel-size=1 --tensor-parallel-size=2 --max-model-len 4096 --gpu-memory-utilization 0.9 --disable-cascade-attn --no-enable-prefix-caching --no-enable_chunked_prefill --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "level": 0}'
```

5. Run client (Input128, Output128, BS1):
```bash
./iluvatar_bench sgl-perf --backend vllm --host 0.0.0.0 --port 8000 --model /path/to/model --dataset-name random --dataset-path /path/to/ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 1 --random-input 128 --max-concurrency 1 --tokenize-prompt --random-range-ratio 1 --random-output 128
```

## References

- [Qwen3](https://github.com/QwenLM/Qwen3)
- [vLLM](https://github.com/vllm-project/vllm)