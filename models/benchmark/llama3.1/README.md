# Llama3.1 测试

* 下载权重和数据集

  FP16权重链接：[meta-llama/Llama-3.1-70B-Instruct · Hugging Face](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct)
  数据集：[HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)

* 模型权重量化

```bash
cd ..
python3 quantize_w8a8.py \
--model /path/to/model \
--dataset-path /home/data/nlp/ultrachat_200k \
--num-samples 32 \
--model-type llama

# w8a8权重保存名称以 -W8A8-Dynamic-Per-Token 结尾
```

* 测试 Llama3.1-70B-instruct W8A8

```bash
cd ..
bash test_performance_server.sh --model /path/to/model -tp 4 --host 127.0.0.1 --port 12345 --max-num-seqs 10 --max-num-batched-tokens 20480 --max-model-len 20480,--model /path/to/model --host 127.0.0.1 --port 12345 --num-prompts 10 --input-tokens 2048 --output-tokens 1024
```

========================================================================================
* 下载权重 Llama3.1-8B-instruct：

     FP16权重链接： meta-llama/Llama-3.1-8B-Instruct · Hugging Face


* 数据集下载并解压： https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip


* 测试 Llama3.1-8B-instruct  bf16/fp16  

```bash
cd ..
bash test_vllm_longbench.sh --model-name llama3.1-8b-chat --model /path/to/model --datapath /path/to/longbench/data -tp 1 --max-model-len 32768 --dtype float16 --max-num-seqs 16 --val-data-nums 1 --temperature 0.0  --max-num-batched-tokens 32768 --trust-remote-code

bash test_vllm_longbench.sh --model-name llama3.1-8b-chat --model /path/to/model --datapath /path/to/longbench/data -tp 1 --max-model-len 32768 --dtype bfloat16 --max-num-seqs 16 --val-data-nums 1 --temperature 0.0  --max-num-batched-tokens 32768 --trust-remote-code

```