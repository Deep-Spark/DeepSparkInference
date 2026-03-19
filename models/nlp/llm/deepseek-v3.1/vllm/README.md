# DeepSeek-V3.1 (vLLM)

## Model Description

DeepSeek-V3 is a powerful Mixture-of-Experts (MoE) language model with 671B total parameters and 37B activated parameters. It achieves excellent performance on math, code, and reasoning tasks, comparable to leading models like GPT-4 and Claude-3.5.

This version supports W4A8 (Weight-4bit, Activation-8bit) quantization for efficient inference.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.4.0 | 26.03 |

## Model Preparation

### Prepare Resources

- Model: <https://www.modelscope.cn/models/deepseek-ai/DeepSeek-V3>

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
export VLLM_W8A8_MOE_USE_W4A8=1
export VLLM_ENFORCE_CUDA_GRAPH=1
export VLLM_PP_LAYER_PARTITION="16,16,16,13"
```

4. Start server:
```bash
vllm serve /path/to/model --trust-remote-code --pipeline-parallel-size=4 --tensor-parallel-size=4 --max-model-len 8192 --gpu-memory-utilization 0.9 --disable-cascade-attn --no-enable-prefix-caching --no-enable_chunked_prefill --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "level": 0}'
```

5. Run client (Input128, Output128, BS8):
```bash
./iluvatar_bench sgl-perf --backend vllm --host 0.0.0.0 --port 8000 --model /path/to/model --dataset-name random --dataset-path /path/to/ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 8 --random-input 128 --max-concurrency 8 --tokenize-prompt --random-range-ratio 1 --random-output 128
```

#### Accuracy Test

6. Install evalscope:
```bash
pip3 install 'evalscope[app,perf]' -U
```

7. Set environment variables:
```bash
export VLLM_USE_MODELSCOPE=True
```

8. Start server:
```bash
vllm serve /path/to/model --max-num-seqs 4 --max-model-len 95600 --served-model-name DeepSeek-v3.1-int4-pack8 --trust-remote-code --disable-cascade-attn --tensor-parallel-size 8 --pipeline-parallel-size 2 --compilation-config '{"level":0,"cudagraph_mode":"FULL_DECODE_ONLY"}' --port 9989
```

9. Run client (MATH-500 dataset):
```bash
evalscope eval --model DeepSeek-v3.1-W4A8 --dataset-args '{"math_500": {"few_shot_num": 0}}'  --generation-config '{"do_sample": true, "temperature": 0.6, "max_tokens": 32768, "n": 1, "top_p": 0.95}' --datasets math_500 --eval-type openai_api --eval-batch-size 4 --api-url http://127.0.0.1:9989/v1 --timeout 12000000  --api-key EMPTY --eval-type openai_api
```

## References

- [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3)
- [vLLM](https://github.com/vllm-project/vllm)