# ChatGLM3-6B (vLLM)

## Model Description

ChatGLM3-6B is trained on large-scale natural language text data, enabling it to understand and generate text. It can be
applied to various natural language processing tasks such as dialogue generation, text summarization, and language
translation.

## Model Preparation

### Prepare Resources

Pretrained model: <https://huggingface.co/THUDM/chatglm3-6b>

```bash
mkdir /data/chatglm/
mv chatglm3-6b.zip/tar /data/chatglm/
```

### Install Dependencies

In order to run the model smoothly, you need to get the sdk from [resource
center](https://support.iluvatar.com/#/ProductLine?id=2) of Iluvatar CoreX official website.

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

pip3 install vllm
pip3 install transformers
```

## Model Inference

```bash
python3 offline_inference.py --model /data/chatglm/chatglm3-6b --trust-remote-code --temperature 0.0 --max-tokens 256
```

### Use the server

Start the server.

```bash
python3 -m vllm.entrypoints.openai.api_server --model /data/chatglm/chatglm3-6b --gpu-memory-utilization 0.9 --max-num-batched-tokens 8193 \
        --max-num-seqs 32 --disable-log-requests --host 127.0.0.1 --port 12345 --trust-remote-code
```

### Test using the OpenAI interface

```bash
python3 server_inference.py --host 127.0.0.1 --port 12345 --model_path /data/chatglm/chatglm3-6b
```

### Benchmarking vLLM

```bash
# Downloading the ShareGPT dataset.
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

# Cloning the vllm project
git clone https://github.com/vllm-project/vllm.git -b v0.5.4 --depth=1
```

Starting server.

```bash
python3 -m vllm.entrypoints.openai.api_server --model /data/chatglm/chatglm3-6b --gpu-memory-utilization 0.9 --max-num-batched-tokens 8193 \
        --max-num-seqs 32 --disable-log-requests --host 127.0.0.1 --trust-remote-code
```

Starting benchmark client.

```bash
python3 benchmark_serving.py --host 127.0.0.1 --num-prompts 16 --model /data/chatglm/chatglm3-6b --dataset-name sharegpt \
        --dataset-path /data/dataset/ShareGPT_V3_unfiltered_cleaned_split.json --sharegpt-output-len 130 --trust-remote-code
```
