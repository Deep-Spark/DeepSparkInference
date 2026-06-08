# Fuyu-8B (vLLM)

## Model Description

Fuyu-8B is a multi-modal text and image transformer trained by Adept AI.

Architecturally, Fuyu is a vanilla decoder-only transformer - there is no image encoder. Image patches are instead
linearly projected into the first layer of the transformer, bypassing the embedding lookup. We simply treat the
transformer decoder like an image transformer (albeit with no pooling and causal attention).

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

- Model: <https://huggingface.co/adept/fuyu-8b>

```bash
cp -r ../../vllm_public_assets/ ./

# Download model from the website and make sure the model's path is "data/fuyu-8b"
mkdir data/
```

### Install Dependencies

Contact the Iluvatar administrator to get the missing packages:
- transformers-*.whl

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx
```

## Model Inference

```bash
export VLLM_ASSETS_CACHE=../vllm/
python3 offline_inference_vision_language.py --model ./data/fuyu-8b --max-tokens 256 -tp 2 --trust-remote-code --temperature 0.0
```

## Model Results

| Model  | QPS | tokens | Token/s |
| :----: | :----: | :----: | :----: |
| fuyu-8b | 0.326   |  17   | 5.379 |