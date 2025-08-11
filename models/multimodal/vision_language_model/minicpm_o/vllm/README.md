# MiniCPM-o 2 (vLLM)

## Model Description

The most capable model in the MiniCPM-o series. With a total of 8B parameters, this end-to-end model achieves comparable
performance to GPT-4o-202405 in vision, speech, and multimodal live streaming, making it one of the most versatile and
performant models in the open-source community. For the new voice mode, MiniCPM-o 2.6 supports bilingual real-time
speech conversation with configurable voices, and also allows for fun capabilities such as emotion/speed/style control,
end-to-end voice cloning, role play, etc. It also advances MiniCPM-V 2.6's visual capabilities such strong OCR
capability, trustworthy behavior, multilingual support, and video understanding. Due to its superior token density,
MiniCPM-o 2.6 can for the first time support multimodal live streaming on end-side devices such as iPad.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.09 |

## Model Preparation

### Prepare Resources

- Model: <https://huggingface.co/openbmb/MiniCPM-o-2_6>

```bash
cp -r ../../vllm_public_assets/ ./
```

### Install Dependencies

Contact the Iluvatar administrator to get the missing packages:

- transformers-4.45.2+corex.4.3.0-py3-none-any.whl

## Model Inference

```bash
export VLLM_ASSETS_CACHE=../vllm/
python3 offline_inference_vision_language.py --model ./MiniCPM-o-2_6/ --max-model-len 4096 --max-num-seqs 2  --trust-remote-code --temperature 0.0 --disable-mm-preprocessor-cache
python3 offline_inference_vision_language.py --model ./MiniCPM-o-2_6/ --max-model-len 4096 --max-num-seqs 2  --trust-remote-code --temperature 0.0 --disable-mm-preprocessor-cache --modality video
```

## Model Results
