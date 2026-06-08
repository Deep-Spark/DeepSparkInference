# Qwen1.5-7B (Text Generation Inference)

## Model Description

Qwen1.5 is a language model series including decoder language models of different model sizes. For each size, we release
the base language model and the aligned chat model. It is based on the Transformer architecture with SwiGLU activation,
attention QKV bias, group query attention, mixture of sliding window attention and full attention, etc. Additionally, we
have an improved tokenizer adaptive to multiple natural languages and codes. For the beta version, temporarily we did
not include GQA (except for 32B) and the mixture of SWA and full attention.

## Supported Environments

| GPU | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release | Branch |
| :----: | :----: | :----: | :----: |
| MR-V100 | 4.4.0 | 26.03 | release/26.03 |
| MR-V100 | 4.3.0 | 25.12 | release/25.12 |

> **Note:** 请切换到与您的 SDK 版本对应的 Release 分支进行测试。请勿直接在 master 分支上运行测试，因为 master 分支可能包含与您的本地 SDK 版本不兼容的最新更改。
>
> 切换分支命令示例：`git checkout release/26.03`

## Model Preparation

### Prepare Resources

- Model: <https://modelscope.cn/models/qwen/Qwen1.5-7B/summary>

```bash
cd ${DeepSparkInference}/models/nlp/llm/qwen1.5-7b/text-generation-inference
mkdir -p data/qwen1.5
ln -s /path/to/Qwen1.5-7B ./data/qwen1.5
```

### Install Dependencies

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx
```

## Model Inference

### Start webserver

```bash
# Use one docker container to start webserver
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
CUDA_VISIBLE_DEVICES=0 USE_FLASH_ATTENTION=true text-generation-launcher --model-id ./data/qwen1.5/Qwen1.5-7B \
                                                                         --sharded false --dtype float16  \
                                                                         --disable-custom-kernels \
                                                                         --port 8001 --max-input-length 2048 \
                                                                         --max-batch-prefill-tokens 2048 \
                                                                         --max-total-tokens 4096 \
                                                                         --max-batch-total-tokens 4096
```

### Offline test

```bash
# Use another docker container to run offline test
export CUDA_VISIBLE_DEVICES=1
python3 offline_inference.py --model2path ./data/qwen1.5/Qwen1.5-7B
```

## Model Results

| Model      | QPS   |
| :----: | :----: |
| Qwen1.5-7B | 39.11 |
