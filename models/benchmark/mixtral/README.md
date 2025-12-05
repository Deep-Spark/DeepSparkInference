# Mixtral 测试

* 下载数据

    联系您的应用工程师获取

* 测试 Mixtral-8x22B-W8A8

```bash
cd ..
bash test_performance_server.sh --model /path/to/model -tp 8 --host 127.0.0.1 --port 12345 --enable-chunked-prefill=False --max-num-seqs 10 --max-num-batched-tokens 20480 --max-model-len 20480,--model /path/to/model --host 127.0.0.1 --port 12345 --num-prompts 10 --input-tokens 2048 --output-tokens 1024
```
