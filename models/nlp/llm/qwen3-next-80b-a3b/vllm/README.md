# Qwen3-Next-80B-A3B-Instruct (vLLM)

## Model Description

Qwen3-Next-80B-A3B-Instruct is a Mixture-of-Experts (MoE) large language model with 80B total parameters and 3B activated parameters. This is the next generation Qwen model with enhanced reasoning capabilities and instruction following.

This version runs in BF16 precision for maximum accuracy.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.4.0 | 26.03 |

## Model Preparation

### Prepare Resources

- Model: <https://www.modelscope.cn/models/Qwen/Qwen3-Next-80B-A3B-Instruct>

### Install Dependencies

In order to run the model smoothly, you need to get the sdk from [resource center](https://support.iluvatar.com/#/ProductLine?id=2) of Iluvatar CoreX official website.

## Model Inference

### Inference with BF16

#### Accuracy Test

1. Install evalscope:
```bash
pip3 install 'evalscope[app,perf]' -U
```

2. Set environment variables:
```bash
export VLLM_USE_MODELSCOPE=True
export VLLM_ENFORCE_CUDA_GRAPH=1
```

3. Start server:
```bash
vllm serve /path/to/model --served-model-name Qwen3-Next-80B-A3B-Instruct --trust_remote_code --port 8801 --pipeline-parallel-size 1 --tensor-parallel-size 8 --max-num-seqs 64 --max-model-len 40960 --disable-cascade-attn --gpu-memory-utilization 0.90 --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "level": 0}' --port 9989
```

4. Run client (MMLU-Pro dataset):
```bash
evalscope eval --model Qwen3-Next-80B-A3B-Instruct --dataset-args '{"mmlu_pro": {"few_shot_num": 0}}'  --generation-config '{"do_sample": true, "temperature": 0.7, "max_tokens": 32768, "n": 1, "top_p": 0.8, "top_k": 20}' --datasets mmlu_pro --eval-type openai_api --eval-batch-size 64 --api-url http://127.0.0.1:9989/v1 --timeout 12000000  --api-key EMPTY --eval-type openai_api
```

## References

- [Qwen3](https://github.com/QwenLM/Qwen3)
- [vLLM](https://github.com/vllm-project/vllm)