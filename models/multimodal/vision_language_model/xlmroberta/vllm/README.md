# XLMRoberta (vLLM)

## Model Description

XLM-RoBERTa is a multilingual version of RoBERTa. It is pre-trained on 2.5TB of filtered CommonCrawl data containing 100 languages.

RoBERTa is a transformers model pretrained on a large corpus in a self-supervised fashion. This means it was pretrained on the raw texts only, with no humans labelling them in any way (which is why it can use lots of publicly available data) with an automatic process to generate inputs and labels from those texts.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.12 |

## Model Preparation

### Prepare Resources

- Model: <https://huggingface.co/BAAI/bge-reranker-v2-m3>
- Model: <https://huggingface.co/intfloat/multilingual-e5-large> base model is xlm-roberta-large

```bash
# Download model from the website and make sure the model's path is "data/bge-reranker-v2-m3" "data/multilingual-e5-large"
mkdir data
```

### Install Dependencies

In order to run the model smoothly, you need to get the sdk from [resource center](https://support.iluvatar.com/#/ProductLine?id=2) of Iluvatar CoreX official website.

## Model Inference

### Sentence Pair Scoring Modeling
```bash
python3 offline_inference_scoring.py --model data/bge-reranker-v2-m3 --task "score" --tensor-parallel-size 1
```

### Text Embedding
```bash
python3 offline_inference_embedding.py --model data/multilingual-e5-large -tp 2
```

## Model Results

| Model  | QPS | tokens | Token/s    |
| :----: | :----: | :----: | :----: |
| multilingual-e5-large | 14.851  |  4096   | 15207.863 |