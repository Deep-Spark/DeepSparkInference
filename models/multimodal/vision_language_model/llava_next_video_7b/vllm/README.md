# LLaVA-Next-Video-7B (vLLM)

## Model Description

LLaVA-Next-Video is an open-source chatbot trained by fine-tuning LLM on multimodal instruction-following data. The
model is buit on top of LLaVa-NeXT by tuning on a mix of video and image data to achieves better video understanding
capabilities. The videos were sampled uniformly to be 32 frames per clip. The model is a current SOTA among open-source
models on VideoMME bench. Base LLM: lmsys/vicuna-7b-v1.5

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

- Model: <https://modelscope.cn/models/swift/LLaVA-NeXT-Video-7B-hf>

```bash
# Download model from the website and make sure the model's path is "data/LLaVA-NeXT-Video-7B-hf"
mkdir data/
```

### Install Dependencies

In order to run the model smoothly, you need to get the sdk from [resource center](https://support.iluvatar.com/#/ProductLine?id=2) of Iluvatar CoreX official website.

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
python3 offline_inference_vision_language.py --model ./data/LLaVA-NeXT-Video-7B-hf --max-tokens 256 -tp 4 --trust-remote-code --temperature 0.0 --model-type llava-next-video --modality video  --dtype bfloat16
```

## Model Results

| Model  | QPS | tokens | Token/s |
| :----: | :----: | :----: | :----: |
| llava | 0.214 |  162   | 34.674 |