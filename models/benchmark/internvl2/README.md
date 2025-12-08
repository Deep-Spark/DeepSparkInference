# InternVL2-26B

* 下载数据

    数据链接: [InternVL2-26B](https://huggingface.co/OpenGVLab/InternVL2-26B)

* 测试 InternVL2-26B

```bash
cd ..
bash test_performance_server_multimodal.sh --model /path/to/model -tp 4 --max-num-seqs 32 --max-num-batched-tokens 8192 --max-model-len 8192 --host 127.0.0.1 --port 12345 --trust-remote-code,--model /path/to/model --host 127.0.0.1 --port 12345 --num-prompts 1 --output-tokens 128 --image-path test.jpg --trust-remote-code true --image-size "512,512"
```
