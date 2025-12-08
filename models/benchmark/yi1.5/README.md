# Yi1.5 测试

* 下载数据

    数据链接: [Yi](https://huggingface.co/01-ai/Yi-1.5-34B-Chat)

* 测试 Yi-1.5-34B-chat

```bash
cd ..
bash test_performance_server.sh --model /path/to/model -tp 4 --host 127.0.0.1 --port 12345 --max-num-batched-tokens 20480 --max-num-seqs 10,--model /path/to/model --host 127.0.0.1 --port 12345 --num-prompts 10 --input-tokens 2048 --output-tokens 1024
```
