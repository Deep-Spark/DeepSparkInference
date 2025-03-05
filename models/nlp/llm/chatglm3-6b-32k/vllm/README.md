# ChatGLM3-6B-32K (vLLM)

## Model Description

ChatGLM3-6B-32K further enhances the understanding of long text capabilities based on ChatGLM3-6B, enabling better
handling of contexts up to 32K in length. Specifically, we have updated the positional encoding and designed more
targeted long text training methods, using a 32K context length during the training phase. In practical use, if your
context length is mostly within 8K, we recommend using ChatGLM3-6B; if you need to handle context lengths exceeding 8K,
we recommend using ChatGLM3-6B-32K.

## Model Preparation

### Prepare Resources

Pretrained model: <https://www.modelscope.cn/models/ZhipuAI/chatglm3-6b-32k>

```bash
mkdir -p /data/chatglm/
mv chatglm3-6b-32k.zip/tar /data/chatglm/
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

pip3 install transformers
```

## Model Inference

```bash
python3 offline_inference.py --model /data/chatglm/chatglm3-6b-32k --trust-remote-code --temperature 0.0 --max-tokens 256
```

### Use the server

Start the server.

```bash
python3 -m vllm.entrypoints.openai.api_server --model /data/chatglm/chatglm3-6b-32k --gpu-memory-utilization 0.9 --max-num-batched-tokens 8193 \
        --max-num-seqs 32 --disable-log-requests --host 127.0.0.1 --port 12345 --trust-remote-code
```

Test using the OpenAI interface.

```bash
python3 server_inference.py --host 127.0.0.1 --port 12345 --model_path /data/chatglm/chatglm3-6b-32k
```

## Model Results

| Model           | Precision | tokens | QPS    |
|-----------------|-----------|--------|--------|
| ChatGLM3-6B-32K | FP16      | 745    | 110.85 |
