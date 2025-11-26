# Qwen-7B (vLLM)

## Model Description

Qwen-7B is a cutting-edge large language model developed as part of the Qwen series, offering advanced natural language
processing capabilities. With 7 billion parameters, it demonstrates exceptional performance across various downstream
tasks. The model comes in two variants: the base pretrained version and the Qwen-Chat version, which is fine-tuned using
human alignment techniques. Notably, Qwen-7B exhibits strong tool-use and planning abilities, making it suitable for
developing intelligent agent applications. It also includes specialized versions for coding (Code-Qwen) and mathematics
(Math-Qwen), showcasing improved performance in these domains compared to other open-source models.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.09 |
| MR-V100 | 4.2.0 | 25.03 |

## Model Preparation

### Prepare Resources

- Model: - Model: <https://modelscope.cn/models/qwen/Qwen-7B/summary>

```bash
cd ${DeepSparkInference}/models/nlp/llm/qwen-7b/vllm
mkdir -p data/qwen
ln -s /path/to/Qwen-7B ./data/qwen
```

### Install Dependencies

In order to run the model smoothly, you need to get the sdk from [resource
center](https://support.iluvatar.com/#/ProductLine?id=2) of Iluvatar CoreX official website.

## Model Inference

```bash
export CUDA_VISIBLE_DEVICES=0,1
python3 offline_inference.py --model ./data/qwen/Qwen-7B-Chat --max-tokens 256 -tp 2 --trust-remote-code --temperature 0.0
```

## Model Results

| Model  | QPS | tokens | Token/s    |
| :----: | :----: | :----: | :----: |
| Qwen-7B-Chat | 0.581  | 603    | 116.919 |