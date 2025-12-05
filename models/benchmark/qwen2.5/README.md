# Qwen2.5 测试

* 下载数据

  数据链接: [Qwen](https://huggingface.co/Qwen)
  数据集：[HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)

* 测试 Qwen2.5-72B-Instruct

```bash
cd ..
bash test_performance_server.sh --model /path/to/model -tp 8 --host 127.0.0.1 --port 12345 --max-num-batched-tokens 20480 --max-num-seqs 10 --max-model-len 20480,--model /path/to/model --host 127.0.0.1 --port 12345 --num-prompts 10 --input-tokens 2048 --output-tokens 1024
```

* 测试 Qwen2.5-72B-instruct W8A8

```bash
## 量化
python3 quantize_w8a8.py \
--model /path/to/model \
--dataset-path /home/data/nlp/ultrachat_200k \
--num-samples 32 \
--model-type qwen

# w8a8权重保存名称以 -W8A8-Dynamic-Per-Token 结尾

## 运行 Qwen2.5-72B-instruct W8A8
cd ..
bash test_performance_server.sh --model /path/to/model -tp 4 --host 127.0.0.1 --port 12345 --max-num-batched-tokens 20480 --max-num-seqs 10 --max-model-len 20480,--model /path/to/model --host 127.0.0.1 --port 12345 --num-prompts 10 --input-tokens 2048 --output-tokens 1024
```

* 测试 Qwen2.5-72B-Instruct-GPTQ-Int4

```bash
cd ..
# tp 2
VLLM_RPC_TIMEOUT=100000 bash test_performance_server.sh --model /path/to/model --quantization gptq -tp 2 --host 127.0.0.1 --port 12345 --max-num-batched-tokens 20480 --max-num-seqs 10 --max-model-len 20480,--model /path/to/model --host 127.0.0.1 --port 12345 --num-prompts 10 --input-tokens 2048 --output-tokens 1024
# tp 4
VLLM_RPC_TIMEOUT=100000 bash test_performance_server.sh --model /path/to/model --quantization gptq -tp 4 --host 127.0.0.1 --port 12345 --max-num-batched-tokens 20480 --max-num-seqs 10 --max-model-len 20480,--model /path/to/model --host 127.0.0.1 --port 12345 --num-prompts 10 --input-tokens 2048 --output-tokens 1024
```

* 测试 Qwen2.5-72B-Instruct-AWQ-Int4

```bash
cd ..
# tp 2
VLLM_RPC_TIMEOUT=100000 bash test_performance_server.sh --model /path/to/model --quantization awq -tp 2 --host 127.0.0.1 --port 12345 --max-num-batched-tokens 20480 --max-num-seqs 10 --max-model-len 20480,--model /path/to/model --host 127.0.0.1 --port 12345 --num-prompts 10 --input-tokens 2048 --output-tokens 1024
# tp 4
VLLM_RPC_TIMEOUT=100000 bash test_performance_server.sh --model /path/to/model --quantization awq -tp 4 --host 127.0.0.1 --port 12345 --max-num-batched-tokens 20480 --max-num-seqs 10 --max-model-len 20480,--model /path/to/model --host 127.0.0.1 --port 12345 --num-prompts 10 --input-tokens 2048 --output-tokens 1024
```

* 测试 Qwen2.5-14B-Instruct性能

```bash
cd ..
bash test_performance_server.sh --model /path/to/model -tp 2 --host 127.0.0.1 --port 12345 --max-num-batched-tokens 20480 --max-num-seqs 10 --max-model-len 20480,--model /path/to/model --host 127.0.0.1 --port 12345 --num-prompts 10 --input-tokens 2048 --output-tokens 1024
```

* 测试 Qwen2.5-14B-Instruct精度

* 数据集下载并解压：https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip

* 测试 Qwen2.5-14B-Instruct  bf16/fp16

```bash
cd ..
bash test_vllm_longbench.sh --model-name qwen2.5-14b-chat --model /path/to/model --datapath /path/to/longbench/data -tp 1 --max-model-len 32768 --dtype float16 --max-num-seqs 8 --val-data-nums 1 --temperature 0.0  --max-num-batched-tokens 32768 --trust-remote-code

bash test_vllm_longbench.sh --model-name qwen2.5-14b-chat --model /path/to/model --datapath /path/to/longbench/data -tp 1 --max-model-len 32768 --dtype bfloat16 --max-num-seqs 8 --val-data-nums 1 --temperature 0.0  --max-num-batched-tokens 32768 --trust-remote-code

```


* 测试Qwen2-VL-7B 性能

```bash
# if "Flash Attention implemented by IXDNN requires last dimension of inputs to be divisible by 32, but get head_dim=80. Optional ENABLE_FLASH_ATTENTION_WITH_HEAD_DIM_PADDING env can be setted to pad along the head dimension." happened.
# set ENABLE_FLASH_ATTENTION_WITH_HEAD_DIM_PADDING=1
cd ..
bash test_performance_server_multimodal.sh --model /path/to/model -tp 1 --max-num-seqs 32 --max-num-batched-tokens 8192 --max-model-len 8192 --host 127.0.0.1 --port 12345,--model /path/to/model --host 127.0.0.1 --port 12345 --num-prompts 1 --output-tokens 128 --image-path test.jpg --image-size "512,512"
```

* 测试Qwen2-VL-7B 精度


* 数据集下载并解压：https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip


* 测试 Qwen2-VL-7B  bf16

```bash
cd ..
bash test_vllm_longbench.sh --model-name qwen2-vl-chat --model /path/to/model --datapath /path/to/longbench/data -tp 1 --max-model-len 32768 --dtype bfloat16 --max-num-seqs 8 --val-data-nums 1 --temperature 0.0  --max-num-batched-tokens 32768 --trust-remote-code

```
