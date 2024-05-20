# ChatGLM3-6B

## Description
ChatGLM3-6B is trained on large-scale natural language text data, enabling it to understand and generate text. It can be applied to various natural language processing tasks such as dialogue generation, text summarization, and language translation. 

## Setup

### Install
In order to run the model smoothly, we need the following dependency files:
1. ixrt-xxx.whl
2. ixformer-xxx.whl
3. vllm-xxx.whl
Please contact the staff to obtain the relevant installation packages.

```bash
yum install mesa-libGL
pip3 install transformers==4.33.2
pip3 install Path/To/ixrt-xxx.whl
pip3 install Path/To/vllm-xxx.whl
pip3 install Path/To/ixformer-xxx.whl
```

### Download
Pretrained model: <https://huggingface.co/THUDM/chatglm3-6b>

```bash
mkdir /data/chatglm/
mv chatglm3-6b.zip/tar /data/chatglm/
```


## Run model

```bash
python3 offline_inference.py --model /data/chatglm/chatglm3-6b --trust-remote-code --temperature 0.0
```

## Use the server

### Start the server

```bash
python3 -m vllm.entrypoints.openai.api_server --model /data/chatglm/chatglm3-6b --gpu-memory-utilization 0.9 --max-num-batched-tokens 8193 \
        --max-num-seqs 32 --disable-log-requests --host 127.0.0.1 --port 12345 --trust-remote-code
```

### Test using the OpenAI interface

```bash
python3 server_inference.py --host 127.0.0.1 --port 12345 --model_path /data/chatglm/chatglm3-6b
```