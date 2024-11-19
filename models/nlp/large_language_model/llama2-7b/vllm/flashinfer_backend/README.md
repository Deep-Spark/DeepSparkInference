# attention 使用不同的backend

通过设置环境变量可以使用不同的attention实现，目前支持两种。
VLLM_ATTENTION_BACKEND=FLASHINFER / XFORMERS(默认使用)

```shell
# offline
VLLM_ATTENTION_BACKEND=FLASHINFER python3 offline_inference.py \
    --model xxx/Meta-Llama-3-8B-Instruct/ \
    --max-tokens 256 \
    --temperature 0.0

# server
VLLM_ATTENTION_BACKEND=FLASHINFER python3 -m vllm.entrypoints.openai.api_server \
    --model xxx/Meta-Llama-3-8B-Instruct/ \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 1024 \
    --host 127.0.0.1 \
    --port 12345
```
