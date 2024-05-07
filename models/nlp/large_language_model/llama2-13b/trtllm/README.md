# Llama2_13b_gpu2

## Description
The Llama2 model is part of the Llama project which aims to unlock the power of large language models. The latest version of the Llama model is now accessible to individuals, creators, researchers, and businesses of all sizes. It includes model weights and starting code for pre-trained and fine-tuned Llama language models with parameters ranging from 7B to 70B. 

## Setup

### Install
In order to run the model smoothly, we need the following dependency files:
1. ixrt-xxx.whl
2. tensorrt_llm-xxx.whl
3. ixformer-xxx.whl
Please contact the staff to obtain the relevant installation packages.

```bash
yum install mesa-libGL
bash set_environment.sh
pip3 install Path/To/ixrt-xxx.whl
pip3 install Path/To/tensorrt_llm-xxx.whl
pip3 install Path/To/ixformer-xxx.whl
```

### Download

Preparing for downloading model
1. To ensure a smooth local deployment of LLaMA2, we first need to apply for permission to download the model files. Currently, there are two places where you can apply: the Meta AI official website and Meta’s model page on HuggingFace.
<https://ai.meta.com/llama/> or <https://huggingface.co/meta-llama>
Regardless of which method you choose, after applying, please wait for a moment. You will receive an email confirming the approval, and then you can refresh the page to proceed with the model download.

2. You can apply for authorization to download the model at the following address:
<https://huggingface.co/llamaste/Llama-2-13b-hf>

3. When the download authorization is approved, you can use the following commands to download the three models according to your needs:
```bash
git lfs install
git clone https://huggingface.co/meta-llama/Llama-2-13b-chat-hf
```

4. After the models you selected finish downloading, you can adjust the directory structure:
```bash
mkdir -p "$PROJECT_DIR/data/llama2"
cd "$PROJECT_DIR/data/llama2"
mv llama-2-13b-chat-hf "$PROJECT_DIR/data/llama2/llama2-13b-chat"
```

Dataset: <https://huggingface.co/datasets/ccdv/cnn_dailymail/tree/main> to download the dataset.
```bash
mv cnn_stories.tgz /Path/To/cnn_dailymail
mv dailymail_stories.tgz /Path/To/cnn_dailymail
python3 download_dateset.py
mkdir -p "$PROJECT_DIR/datasets"
cd "$PROJECT_DIR/datasets"
mv datasets_cnn_dailymail "$PROJECT_DIR/datasets"
```

## Inference
```bash
export CUDA_VISIBLE_DEVICES=DEVICE_1,DEVICE_2
export TLLM_LOG_LEVEL=info
export PLUGIN_DTYPE="float16"
export BS=${BS:-1}
export DTYPE=${DTYPE:-"float16"}
export PROJECT_DIR=/PATH/To/llm
export DATASET_DIR=${DATASET_DIR:-"${PROJECT_DIR}/datasets/datasets_cnn_dailymail"}
export MODEL_DIR=${MODEL_DIR:-"${PROJECT_DIR}/data/llama2/llama2-13b-chat"}
export ENGINE_DIR=${ENGINE_DIR:-"${PROJECT_DIR}/tmp/trtllm/llama2/13B/trt_engines/fp16/2-gpu/"}
```
### FP16

```bash
# Build Engine
bash scripts/test_trtllm_llama2_13b_gpu2_build.sh
# summarize
bash scripts/test_trtllm_llama2_13b_gpu2.sh
```
