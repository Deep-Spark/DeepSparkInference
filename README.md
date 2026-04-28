<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
[<img alt="English" src="https://img.shields.io/badge/Language-English-blue.svg">](README_en.md) [<img alt="Chinese" src="https://img.shields.io/badge/语言-简体中文-red.svg">](README.md)

# DeepSparkInference

<div align="center" style="line-height: 1;">
  <a href="https://www.deepspark.org.cn"><img alt="Homepage"
    src="https://img.shields.io/badge/DeepSpark-Homepage-blue.svg"/></a>
  <a href="./LICENSE"><img alt="LICENSE" src="https://img.shields.io/badge/license-Apache%202.0-dfd.svg"></a>
  <a href="https://gitee.com/deep-spark/deepsparkinference/releases/latest"><img alt="Release" src="https://img.shields.io/github/v/release/deep-spark/deepsparkinference?color=ffa"></a>
</div>
<br>

## 近期活动

* 🔥天数智芯 x 百度飞桨黑客松活动火热来袭(3月27日～6月16日）！🔥
  * [报名入口](https://github.com/PaddlePaddle/Paddle/issues/78485)
  * [打卡任务](https://github.com/PaddlePaddle/community/blob/master/pfcc/paddle-hardware/%E5%9F%BA%E4%BA%8E%E5%A4%A9%E6%95%B0%E6%99%BA%E8%8A%AF-BI-150S-%E7%9A%84-FastDeploy-%E7%BC%96%E8%AF%91%E4%B8%8E%E6%89%93%E5%8C%85.md)
  * [进阶任务](https://github.com/PaddlePaddle/community/blob/master/hackathon/hackathon_10th/%E3%80%90Hackathon_10th%E3%80%91%E6%96%87%E5%BF%83%E5%90%88%E4%BD%9C%E4%BC%99%E4%BC%B4%E4%BB%BB%E5%8A%A1%E5%90%88%E9%9B%86.md#%E5%A4%A9%E6%95%B0%E6%99%BA%E8%8A%AF%E5%9F%BA%E4%BA%8E%E5%A4%A9%E6%95%B0%E6%99%BA%E8%8A%AF%E7%A1%AC%E4%BB%B6%E4%B8%8E%E6%96%87%E5%BF%83%E5%A4%9A%E6%A8%A1%E6%80%81%E6%A8%A1%E5%9E%8B%E7%9A%84%E5%88%9B%E6%96%B0%E5%BA%94%E7%94%A8)
  * [算力指南](https://aistudio.baidu.com/projectdetail/10039684)

赛道 |说明	| 奖励
---|---|---
打卡任务 |	完成打卡任务即可获得周边奖励，长期开放 |	百度周边礼品 + 伙伴周边礼品，前10名完成者可获得 100 元价值礼品，所有完赛选手有电子证书
进阶任务 |	完成打卡任务评审后可认领进阶任务，厂商伙伴会进行RFC或者面试 |	单人奖金 2000 元 * 2人 / 进阶任务
开源贡献任务 |	部分合作伙伴特别设立「开源贡献奖励」，鼓励开发者在参赛过程中积极发现问题、提出建议（ISSUE）、贡献代码（PR）|	每一次高质量贡献都能获得额外奖励！

除了FastDeploy外，更多推理模型实例请参看本项目，已有300个大小模型示例，欢迎体验！

## 项目介绍

`DeepSparkInference`推理模型库作为 DeepSpark 开源社区的核心项目，于 2024 年 3 月正式开源，一期甄选了 48 个推理模型示例，涵盖计算机视觉，自然语言处理，语音识别等领域，后续将逐步拓展更多 AI 领域。

`DeepSparkInference`中的模型提供了在国产推理引擎`IGIE`或`ixRT`下运行的推理示例和指导文档，部分模型提供了基于国产通用 GPU [智铠 100](https://www.iluvatar.com/productDetails?fullCode=cpjs-yj-tlxltt-zk100) 的评测结果。

`IGIE`（Iluvatar GPU Inference Engine）是基于 TVM 框架研发的高性能、高通用、全流程的 AI 推理引擎。支持多框架模型导入、量化、图优化、多算子库支持、多后端支持、算子自动调优等特性，为推理场景提供易部署、高吞吐量、低延迟的完整方案。

`ixRT`（Iluvatar CoreX RunTime）是天数智芯自研的高性能推理引擎，专注于最大限度发挥天数智芯通用 GPU 的性能，实现各领域模型的高性能推理。`ixRT`支持动态形状推理、插件和 INT8/FP16 推理等特性。

`DeepSparkInference`将按季度进行版本更新，后续会逐步丰富模型类别并拓展大模型推理。

## 模型库

### 大语言模型（LLM）

| Model                         | Engine       | Supported                                                          | IXUCA SDK |
|-------------------------------|--------------|--------------------------------------------------------------------|-----------|
| Baichuan2-7B                  | `vLLM`       | [✅](models/nlp/llm/baichuan2-7b/vllm)                              | 4.3.0     |
| ChatGLM-3-6B                  | `vLLM`       | [✅](models/nlp/llm/chatglm3-6b/vllm)                               | 4.3.0     |
| ChatGLM-3-6B-32K              | `vLLM`       | [✅](models/nlp/llm/chatglm3-6b-32k/vllm)                           | 4.3.0     |
| CosyVoice2-0.5B               | `PyTorch`    | [✅](models/speech/speech_synthesis/cosyvoice/pytorch)              | 4.3.0     |
| CosyVoice2-0.5B               | `ixRT`       | [✅](models/speech/speech_synthesis/cosyvoice/ixrt)                 | dev-only  |
| DeepSeek-R1-Distill-Llama-8B  | `vLLM`       | [✅](models/nlp/llm/deepseek-r1-distill-llama-8b/vllm)              | 4.3.0     |
| DeepSeek-R1-Distill-Llama-70B | `vLLM`       | [✅](models/nlp/llm/deepseek-r1-distill-llama-70b/vllm)             | 4.3.0     |
| DeepSeek-R1-Distill-Qwen-1.5B | `vLLM`       | [✅](models/nlp/llm/deepseek-r1-distill-qwen-1.5b/vllm)             | 4.3.0     |
| DeepSeek-R1-Distill-Qwen-7B   | `vLLM`       | [✅](models/nlp/llm/deepseek-r1-distill-qwen-7b/vllm)               | 4.4.0     |
| DeepSeek-R1-Distill-Qwen-14B  | `vLLM`       | [✅](models/nlp/llm/deepseek-r1-distill-qwen-14b/vllm)              | 4.3.0     |
| DeepSeek-R1-Distill-Qwen-32B  | `vLLM`       | [✅](models/nlp/llm/deepseek-r1-distill-qwen-32b/vllm)              | 4.3.0     |
| DeepSeek-OCR                  | `Transformers` | [✅](models/multimodal/vision_language_model/deepseek-ocr/transformers)  | 4.3.0 |
| DeepSeek-OCR                  | `vLLM`       | [✅](models/multimodal/vision_language_model/deepseek-ocr/vllm)     | dev-only  |
| ERNIE-4.5-21B-A3B             | `FastDeploy` | [✅](models/nlp/llm/ernie-4.5-21b-a3b/fastdeploy/)                  | 4.3.0     |
| ERNIE-4.5-300B-A47B           | `FastDeploy` | [✅](models/nlp/llm/ernie-4.5-300b-a47b/fastdeploy/)                | 4.3.0     |
| ERNIE-4.5-VL-28B-A3B-Thinking | `Transformers` | [✅](models/multimodal/vision_language_model/ernie-4.5-vl-28b-a3b-thinking/transformers/) | 4.4.0     |
| GLM-4V                        | `vLLM`       | [✅](models/multimodal/vision_language_model/glm-4v/vllm/)          | 4.3.0     |
| InternLM3                     | `LMDeploy`   | [✅](models/nlp/llm/internlm3/lmdeploy/)                            | 4.3.0     |
| InternLM3                     | `vLLM`       | [✅](models/nlp/llm/internlm3/vllm/)                                | 4.4.0     |
| IndexTTS-2                    | `IndexTTS`       | [✅](models/speech/speech_synthesis/indextts-2/indextts/)       | 4.4.0     |
| Llama2-7B                     | `vLLM`       | [✅](models/nlp/llm/llama2-7b/vllm)                                 | 4.3.0     |
| Llama2-7B                     | `TRT-LLM`    | [✅](models/nlp/llm/llama2-7b/trtllm)                               | 4.3.0     |
| Llama2-13B                    | `TRT-LLM`    | [✅](models/nlp/llm/llama2-13b/trtllm)                              | 4.3.0     |
| Llama2-70B                    | `TRT-LLM`    | [✅](models/nlp/llm/llama2-70b/trtllm)                              | 4.3.0     |
| Llama3-70B                    | `vLLM`       | [✅](models/nlp/llm/llama3-70b/vllm)                                | 4.3.0     |
| E5-V                          | `vLLM`       | [✅](models/multimodal/vision_language_model/e5-v/vllm/)            | 4.3.0     |
| MiniCPM-o-2                   | `vLLM`       | [✅](models/multimodal/vision_language_model/minicpm-o-2/vllm/)     | 4.3.0     |
| MiniCPM-V-2                   | `vLLM`       | [✅](models/multimodal/vision_language_model/minicpm-v-2/vllm/)     | 4.3.0     |
| MiniCPM-V-4                   | `vLLM`       | [✅](models/multimodal/vision_language_model/minicpm-v-4/vllm/)     | dev-only  |
| NVLM                          | `vLLM`       | [✅](models/multimodal/vision_language_model/nvlm/vllm)             | 4.3.0     |
| Phi3_v                        | `vLLM`       | [✅](models/multimodal/vision_language_model/phi3_v/vllm)           | 4.3.0     |
| PaliGemma                     | `vLLM`       | [✅](models/multimodal/vision_language_model/paligemma/vllm)        | 4.3.0     |
| PaddleOCR-VL                  | `Transformers` | [✅](models/multimodal/vision_language_model/paddleocr-vl-1.5/transformers)  | 4.4.0 |
| Qwen-7B                       | `vLLM`       | [✅](models/nlp/llm/qwen-7b/vllm)                                   | 4.3.0     |
| Qwen-VL                       | `vLLM`       | [✅](models/multimodal/vision_language_model/qwen_vl/vllm)          | 4.3.0     |
| Qwen2-VL                      | `vLLM`       | [✅](models/multimodal/vision_language_model/qwen2_vl/vllm)         | 4.3.0     |
| Qwen2.5-VL                    | `vLLM`       | [✅](models/multimodal/vision_language_model/qwen2_5_vl/vllm)       | 4.4.0     |
| Qwen1.5-7B                    | `vLLM`       | [✅](models/nlp/llm/qwen1.5-7b/vllm)                                | 4.3.0     |
| Qwen1.5-7B                    | `TGI`        | [✅](models/nlp/llm/qwen1.5-7b/tgi)                                 | 4.3.0     |
| Qwen1.5-14B                   | `vLLM`       | [✅](models/nlp/llm/qwen1.5-14b/vllm)                               | 4.3.0     |
| Qwen1.5-32B Chat              | `vLLM`       | [✅](models/nlp/llm/qwen1.5-32b/vllm)                               | 4.3.0     |
| Qwen1.5-72B                   | `vLLM`       | [✅](models/nlp/llm/qwen1.5-72b/vllm)                               | 4.3.0     |
| Qwen2-7B Instruct             | `vLLM`       | [✅](models/nlp/llm/qwen2-7b/vllm)                                  | 4.3.0     |
| Qwen2-72B Instruct            | `vLLM`       | [✅](models/nlp/llm/qwen2-72b/vllm)                                 | 4.3.0     |
| Qwen3_Moe                     | `vLLM`       | [✅](models/nlp/llm/qwen3-235b/vllm)                                | dev-only  |
| Qwen3-8B                      | `vLLM`       | [✅](models/nlp/llm/qwen3-8b/vllm)                                  | 4.4.0     |
| Qwen3-32B                     | `vLLM`       | [✅](models/nlp/llm/qwen3-32b/vllm)                                 | 4.4.0     |
| Qwen3-30B-A3B-Thinking        | `vLLM`       | [✅](models/nlp/llm/qwen3-30b-a3b-thinking/vllm)                   | 4.4.0     |
| Qwen3-235B-A22B-Thinking      | `vLLM`       | [✅](models/nlp/llm/qwen3-235b-a22b-thinking/vllm)                 | 4.4.0     |
| Qwen3-Next-80B-A3B            | `vLLM`       | [✅](models/nlp/llm/qwen3-next-80b-a3b/vllm)                       | 4.4.0     |
| Qwen3-Embedding-8B            | `vLLM`       | [✅](models/nlp/llm/qwen3-embedding-8b/vllm)                       | 4.4.0     |
| Qwen3-ASR-1.7B                | `Qwen-ASR`   | [✅](models/speech/asr/qwen3-embedding-8b/qwen-asr)                | 4.4.0     |
| Qwen3-TTS-12Hz-1.7B-Base      | `Qwen-TTS`   | [✅](models/speech/speech_synthesis/qwen3-tts-12hz-1.7b/qwen-tts)   | 4.4.0     |
| DeepSeek-V3.1                 | `vLLM`       | [✅](models/nlp/llm/deepseek-v3.1/vllm)                            | 4.4.0     |
| StableLM2-1.6B                | `vLLM`       | [✅](models/nlp/llm/stablelm/vllm)                                  | 4.3.0     |
| Step3                         | `vLLM`       | [✅](models/multimodal/vision_language_model/step3/vllm)            | 4.4.0     |
| Ultravox                      | `vLLM`       | [✅](models/speech/asr/ultravox/vllm)                               | 4.3.0     |
| Whisper                       | `vLLM`       | [✅](models/speech/asr/whisper/vllm/)                               | 4.3.0     |
| XLMRoberta                    | `vLLM`       | [✅](models/multimodal/vision_language_model/xlmroberta/vllm)       | 4.3.0     |

### 计算机视觉（CV）

#### 视觉分类

| Model                  | Prec. | IGIE                                                   | ixRT                                                      | IXUCA SDK |
|------------------------|-------|--------------------------------------------------------|-----------------------------------------------------------|-----------|
| AlexNet                | FP16  | [✅](models/cv/classification/alexnet/igie)             | [✅](models/cv/classification/alexnet/ixrt)                | 4.3.0     |
|                        | INT8  | [✅](models/cv/classification/alexnet/igie)             | [✅](models/cv/classification/alexnet/ixrt)                | 4.3.0     |
| CLIP                   | FP16  | [✅](models/cv/classification/clip/igie)                | [✅](models/cv/classification/clip/ixrt)                   | 4.3.0     |
| Conformer-B            | FP16  | [✅](models/cv/classification/conformer_base/igie)      |                                                           | 4.3.0     |
| ConvNeXt-Base          | FP16  | [✅](models/cv/classification/convnext_base/igie)       | [✅](models/cv/classification/convnext_base/ixrt)          | 4.3.0     |
| ConvNext-S             | FP16  | [✅](models/cv/classification/convnext_s/igie)          |                                                           | 4.3.0     |
| ConvNeXt-Small         | FP16  | [✅](models/cv/classification/convnext_small/igie)      | [✅](models/cv/classification/convnext_small/ixrt)         | 4.3.0     |
| ConvNeXt-Tiny          | FP16  | [✅](models/cv/classification/convnext_tiny/igie)       |                                                           | 4.3.0     |
| CSPDarkNet53           | FP16  | [✅](models/cv/classification/cspdarknet53/igie)        | [✅](models/cv/classification/cspdarknet53/ixrt)           | 4.3.0     |
|                        | INT8  |                                                        | [✅](models/cv/classification/cspdarknet53/ixrt)           | 4.3.0     |
| CSPResNet50            | FP16  | [✅](models/cv/classification/cspresnet50/igie)         | [✅](models/cv/classification/cspresnet50/ixrt)            | 4.3.0     |
|                        | INT8  |                                                        | [✅](models/cv/classification/cspresnet50/ixrt)            | 4.3.0     |
| CSPResNeXt50           | FP16  | [✅](models/cv/classification/cspresnext50/igie)        | [✅](models/cv/classification/cspresnext50/ixrt)           | 4.3.0     |
| DeiT-B                 | FP16  |                                                          | [✅](models/cv/classification/deit_b/ixrt)              | 4.4.0     |
| DeiT-tiny              | FP16  | [✅](models/cv/classification/deit_tiny/igie)           | [✅](models/cv/classification/deit_tiny/ixrt)              | 4.3.0     |
| DenseNet121            | FP16  | [✅](models/cv/classification/densenet121/igie)         | [✅](models/cv/classification/densenet121/ixrt)            | 4.3.0     |
|                        | INT8  | [✅](models/cv/classification/densenet121/igie)         |                                                            | 4.4.0     |
| DenseNet161            | FP16  | [✅](models/cv/classification/densenet161/igie)         | [✅](models/cv/classification/densenet161/ixrt)            | 4.3.0     |
| DenseNet169            | FP16  | [✅](models/cv/classification/densenet169/igie)         | [✅](models/cv/classification/densenet169/ixrt)            | 4.3.0     |
| DenseNet201            | FP16  | [✅](models/cv/classification/densenet201/igie)         | [✅](models/cv/classification/densenet201/ixrt)            | 4.3.0     |
| EfficientNet-B0        | FP16  | [✅](models/cv/classification/efficientnet_b0/igie)     | [✅](models/cv/classification/efficientnet_b0/ixrt)        | 4.3.0     |
|                        | INT8  |                                                        | [✅](models/cv/classification/efficientnet_b0/ixrt)        | 4.3.0     |
| EfficientNet-B1        | FP16  | [✅](models/cv/classification/efficientnet_b1/igie)     | [✅](models/cv/classification/efficientnet_b1/ixrt)        | 4.3.0     |
|                        | INT8  |                                                        | [✅](models/cv/classification/efficientnet_b1/ixrt)        | 4.3.0     |
| EfficientNet-B2        | FP16  | [✅](models/cv/classification/efficientnet_b2/igie)     | [✅](models/cv/classification/efficientnet_b2/ixrt)        | 4.3.0     |
| EfficientNet-B3        | FP16  | [✅](models/cv/classification/efficientnet_b3/igie)     | [✅](models/cv/classification/efficientnet_b3/ixrt)        | 4.3.0     |
| EfficientNet-B4        | FP16  | [✅](models/cv/classification/efficientnet_b4/igie)     | [✅](models/cv/classification/efficientnet_b4/ixrt)        | 4.3.0     |
| EfficientNet-B5        | FP16  | [✅](models/cv/classification/efficientnet_b5/igie)     | [✅](models/cv/classification/efficientnet_b5/ixrt)        | 4.3.0     |
| EfficientNet-B6        | FP16  | [✅](models/cv/classification/efficientnet_b6/igie)     |                                                           | 4.3.0     |
| EfficientNet-B7        | FP16  | [✅](models/cv/classification/efficientnet_b7/igie)     |                                                           | 4.3.0     |
| EfficientNetV2         | FP16  | [✅](models/cv/classification/efficientnet_v2/igie)     | [✅](models/cv/classification/efficientnet_v2/ixrt)        | 4.3.0     |
|                        | INT8  |                                                        | [✅](models/cv/classification/efficientnet_v2/ixrt)        | 4.3.0     |
| EfficientNetv2_rw_t    | FP16  | [✅](models/cv/classification/efficientnetv2_rw_t/igie) | [✅](models/cv/classification/efficientnetv2_rw_t/ixrt)    | 4.3.0     |
| EfficientNetv2_s       | FP16  | [✅](models/cv/classification/efficientnet_v2_s/igie)   | [✅](models/cv/classification/efficientnet_v2_s/ixrt)      | 4.3.0     |
| GoogLeNet              | FP16  | [✅](models/cv/classification/googlenet/igie)           | [✅](models/cv/classification/googlenet/ixrt)              | 4.3.0     |
|                        | INT8  | [✅](models/cv/classification/googlenet/igie)           | [✅](models/cv/classification/googlenet/ixrt)              | 4.3.0     |
| HRNet-W18              | FP16  | [✅](models/cv/classification/hrnet_w18/igie)           | [✅](models/cv/classification/hrnet_w18/ixrt)              | 4.3.0     |
|                        | INT8  |                                                        | [✅](models/cv/classification/hrnet_w18/ixrt)              | 4.3.0     |
| InceptionV3            | FP16  | [✅](models/cv/classification/inception_v3/igie)        | [✅](models/cv/classification/inception_v3/ixrt)           | 4.3.0     |
|                        | INT8  | [✅](models/cv/classification/inception_v3/igie)        | [✅](models/cv/classification/inception_v3/ixrt)           | 4.3.0     |
| Inception-ResNet-V2    | FP16  |                                                        | [✅](models/cv/classification/inception_resnet_v2/ixrt)    | 4.3.0     |
|                        | INT8  |                                                        | [✅](models/cv/classification/inception_resnet_v2/ixrt)    | 4.3.0     |
| Mixer_B                | FP16  | [✅](models/cv/classification/mlp_mixer_base/igie)      |                                                           | 4.3.0     |
| MNASNet0_5             | FP16  | [✅](models/cv/classification/mnasnet0_5/igie)          |                                                           | 4.3.0     |
| MNASNet0_75            | FP16  | [✅](models/cv/classification/mnasnet0_75/igie)         |                                                           | 4.3.0     |
| MNASNet1_0             | FP16  | [✅](models/cv/classification/mnasnet1_0/igie)          |                                                           | 4.3.0     |
| MNASNet1_3             | FP16  | [✅](models/cv/classification/mnasnet1_3/igie)          |                                                           | 4.3.0     |
| MobileNetV1            | FP16  |                                                        | [✅](models/cv/classification/mobilenet_v1/ixrt)           | 4.4.0     |
| MobileNetV2            | FP16  | [✅](models/cv/classification/mobilenet_v2/igie)        | [✅](models/cv/classification/mobilenet_v2/ixrt)           | 4.3.0     |
|                        | INT8  | [✅](models/cv/classification/mobilenet_v2/igie)        | [✅](models/cv/classification/mobilenet_v2/ixrt)           | 4.3.0     |
| MobileNetV3_Large      | FP16  | [✅](models/cv/classification/mobilenet_v3_large/igie)  |                                                           | 4.3.0     |
| MobileNetV3_Small      | FP16  | [✅](models/cv/classification/mobilenet_v3/igie)        | [✅](models/cv/classification/mobilenet_v3/ixrt)           | 4.3.0     |
| Mobilevit_s            | FP16  | [✅](models/cv/classification/mobilevit_s/igie)         |                                                             | 4.4.0     |
| MViTv2_base            | FP16  | [✅](models/cv/classification/mvitv2_base/igie)         |                                                           | dev-only   |
| RegNet_x_16gf          | FP16  | [✅](models/cv/classification/regnet_x_16gf/igie)       |                                                           | 4.3.0     |
| RegNet_x_1_6gf         | FP16  | [✅](models/cv/classification/regnet_x_1_6gf/igie)      |                                                           | 4.3.0     |
| RegNet_x_3_2gf         | FP16  | [✅](models/cv/classification/regnet_x_3_2gf/igie)      |                                                           | 4.3.0     |
| RegNet_x_8gf          | FP16  | [✅](models/cv/classification/regnet_x_8gf/igie)         |                                                           | 4.3.0     |
| RegNet_x_32gf          | FP16  | [✅](models/cv/classification/regnet_x_32gf/igie)       |                                                           | 4.3.0     |
| RegNet_x_400mf         | FP16  | [✅](models/cv/classification/regnet_x_400mf/igie)      |                                                           | 4.3.0     |
| RegNet_x_800mf         | FP16  | [✅](models/cv/classification/regnet_x_800mf/igie)      |                                                           | 4.3.0     |
| RegNet_y_1_6gf         | FP16  | [✅](models/cv/classification/regnet_y_1_6gf/igie)      |                                                           | 4.3.0     |
| RegNet_y_16gf          | FP16  | [✅](models/cv/classification/regnet_y_16gf/igie)       |                                                           | 4.3.0     |
| RegNet_y_3_2gf         | FP16  | [✅](models/cv/classification/regnet_y_3_2gf/igie)      |                                                           | 4.3.0     |
| RegNet_y_32gf          | FP16  | [✅](models/cv/classification/regnet_y_32gf/igie)       |                                                           | 4.3.0     |
| RegNet_y_400mf         | FP16  | [✅](models/cv/classification/regnet_y_400mf/igie)      |                                                           | 4.3.0     |
| RepVGG                 | FP16  | [✅](models/cv/classification/repvgg/igie)              | [✅](models/cv/classification/repvgg/ixrt)                 | 4.3.0     |
| Res2Net50              | FP16  | [✅](models/cv/classification/res2net50/igie)           | [✅](models/cv/classification/res2net50/ixrt)              | 4.3.0     |
|                        | INT8  |                                                        | [✅](models/cv/classification/res2net50/ixrt)              | 4.3.0     |
| ResNeSt50              | FP16  | [✅](models/cv/classification/resnest50/igie)           |                                                           | 4.3.0     |
| ResNet101              | FP16  | [✅](models/cv/classification/resnet101/igie)           | [✅](models/cv/classification/resnet101/ixrt)              | 4.3.0     |
|                        | INT8  | [✅](models/cv/classification/resnet101/igie)           | [✅](models/cv/classification/resnet101/ixrt)              | 4.3.0     |
| ResNet152              | FP16  | [✅](models/cv/classification/resnet152/igie)           |                                                           | 4.3.0     |
|                        | INT8  | [✅](models/cv/classification/resnet152/igie)           |                                                           | 4.3.0     |
| ResNet18               | FP16  | [✅](models/cv/classification/resnet18/igie)            | [✅](models/cv/classification/resnet18/ixrt)               | 4.3.0     |
|                        | INT8  | [✅](models/cv/classification/resnet18/igie)            | [✅](models/cv/classification/resnet18/ixrt)               | 4.3.0     |
| ResNet34               | FP16  |                                                        | [✅](models/cv/classification/resnet34/ixrt)               | 4.3.0     |
|                        | INT8  |                                                        | [✅](models/cv/classification/resnet34/ixrt)               | 4.3.0     |
| ResNet50               | FP16  | [✅](models/cv/classification/resnet50/igie)            | [✅](models/cv/classification/resnet50/ixrt)               | 4.3.0     |
|                        | INT8  | [✅](models/cv/classification/resnet50/igie)            |                                                           | 4.3.0     |
| ResNetV1D50            | FP16  | [✅](models/cv/classification/resnetv1d50/igie)         | [✅](models/cv/classification/resnetv1d50/ixrt)            | 4.3.0     |
|                        | INT8  |                                                        | [✅](models/cv/classification/resnetv1d50/ixrt)            | 4.3.0     |
| ResNeXt50_32x4d        | FP16  | [✅](models/cv/classification/resnext50_32x4d/igie)     | [✅](models/cv/classification/resnext50_32x4d/ixrt)        | 4.3.0     |
| ResNeXt101_64x4d       | FP16  | [✅](models/cv/classification/resnext101_64x4d/igie)    | [✅](models/cv/classification/resnext101_64x4d/ixrt)       | 4.3.0     |
| ResNeXt101_32x8d       | FP16  | [✅](models/cv/classification/resnext101_32x8d/igie)    | [✅](models/cv/classification/resnext101_32x8d/ixrt)       | 4.3.0     |
| SEResNet50             | FP16  | [✅](models/cv/classification/se_resnet50/igie)         |                                                           | 4.3.0     |
| ShuffleNetV1           | FP16  |                                                        | [✅](models/cv/classification/shufflenet_v1/ixrt)          | 4.3.0     |
| ShuffleNetV2_x0_5      | FP16  | [✅](models/cv/classification/shufflenetv2_x0_5/igie)   | [✅](models/cv/classification/shufflenetv2_x0_5/ixrt)      | 4.3.0     |
| ShuffleNetV2_x1_0      | FP16  | [✅](models/cv/classification/shufflenetv2_x1_0/igie)   | [✅](models/cv/classification/shufflenetv2_x1_0/ixrt)      | 4.3.0     |
| ShuffleNetV2_x1_5      | FP16  | [✅](models/cv/classification/shufflenetv2_x1_5/igie)   | [✅](models/cv/classification/shufflenetv2_x1_5/ixrt)      | 4.3.0     |
| ShuffleNetV2_x2_0      | FP16  | [✅](models/cv/classification/shufflenetv2_x2_0/igie)   | [✅](models/cv/classification/shufflenetv2_x2_0/ixrt)      | 4.3.0     |
| SqueezeNet 1.0         | FP16  | [✅](models/cv/classification/squeezenet_v1_0/igie)     | [✅](models/cv/classification/squeezenet_v1_0/ixrt)        | 4.3.0     |
|                        | INT8  |                                                        | [✅](models/cv/classification/squeezenet_v1_0/ixrt)        | 4.3.0     |
| SqueezeNet 1.1         | FP16  | [✅](models/cv/classification/squeezenet_v1_1/igie)     | [✅](models/cv/classification/squeezenet_v1_1/ixrt)        | 4.3.0     |
|                        | INT8  |                                                        | [✅](models/cv/classification/squeezenet_v1_1/ixrt)        | 4.3.0     |
| SVT Base               | FP16  | [✅](models/cv/classification/svt_base/igie)            |                                                           | 4.3.0     |
| Swin Transformer       | FP16  | [✅](models/cv/classification/swin_transformer/igie)    | [✅](models/cv/classification/swin_transformer/ixrt)     | 4.3.0     |
| Swin Transformer Large | FP16  |                                                        | [✅](models/cv/classification/swin_transformer_large/ixrt) | 4.3.0     |
| Twins_PCPVT            | FP16  | [✅](models/cv/classification/twins_pcpvt/igie)         |                                                           | 4.3.0     |
| VAN_B0                 | FP16  | [✅](models/cv/classification/van_b0/igie)              |                                                           | 4.3.0     |
| VGG11                  | FP16  | [✅](models/cv/classification/vgg11/igie)               |                                                           | 4.3.0     |
| VGG13                  | FP16  | [✅](models/cv/classification/vgg13/igie)               |                                                           | 4.3.0     |
| VGG13_BN               | FP16  | [✅](models/cv/classification/vgg13_bn/igie)            |                                                           | 4.3.0     |
| VGG16                  | FP16  | [✅](models/cv/classification/vgg16/igie)               | [✅](models/cv/classification/vgg16/ixrt)                  | 4.3.0     |
|                        | INT8  | [✅](models/cv/classification/vgg16/igie)               |                                                           | 4.3.0     |
| VGG19                  | FP16  | [✅](models/cv/classification/vgg19/igie)               |                                                           | 4.3.0     |
| VGG19_BN               | FP16  | [✅](models/cv/classification/vgg19_bn/igie)            |                                                           | 4.3.0     |
| ViT                    | FP16  | [✅](models/cv/classification/vit/igie)                 | [✅](models/cv/classification/vit/ixit)                 | 4.3.0     |
| ViT-B-32               | FP16  | [✅](models/cv/classification/vit_b_32/igie)            |                                                           | 4.4.0     |
| ViT-L-14               | FP16  | [✅](models/cv/classification/vit_l_14/igie)            |                                                           | 4.4.0     |
| Wide ResNet50          | FP16  | [✅](models/cv/classification/wide_resnet50/igie)       | [✅](models/cv/classification/wide_resnet50/ixrt)          | 4.3.0     |
|                        | INT8  | [✅](models/cv/classification/wide_resnet50/igie)       | [✅](models/cv/classification/wide_resnet50/ixrt)          | 4.3.0     |
| Wide ResNet101         | FP16  | [✅](models/cv/classification/wide_resnet101/igie)      |                                                           | 4.3.0     |

#### 目标检测

| Model      | Prec. | IGIE                                            | ixRT                                            | IXUCA SDK |
|------------|-------|-------------------------------------------------|-------------------------------------------------|-----------|
| ATSS       | FP16  | [✅](models/cv/object_detection/atss/igie)       | [✅](models/cv/object_detection/atss/ixrt)       | 4.3.0     |
| CenterNet  | FP16  | [✅](models/cv/object_detection/centernet/igie)  | [✅](models/cv/object_detection/centernet/ixrt)  | 4.3.0     |
| DETR       | FP16  | [✅](models/cv/object_detection/detr/igie)       | [✅](models/cv/object_detection/detr/ixrt)       | 4.3.0     |
| FCOS       | FP16  | [✅](models/cv/object_detection/fcos/igie)       | [✅](models/cv/object_detection/fcos/ixrt)       | 4.3.0     |
| FoveaBox   | FP16  | [✅](models/cv/object_detection/foveabox/igie)   | [✅](models/cv/object_detection/foveabox/ixrt)   | 4.3.0     |
| FSAF       | FP16  | [✅](models/cv/object_detection/fsaf/igie)       | [✅](models/cv/object_detection/fsaf/ixrt)       | 4.3.0     |
| GFL        | FP16  | [✅](models/cv/object_detection/gfl/igie)        |                                                 | 4.3.0     |
| Grounding DINO | FP16  |                                               | [✅](models/cv/object_detection/grounding_dino/ixrt) | dev-only |
| HRNet      | FP16  | [✅](models/cv/object_detection/hrnet/igie)      | [✅](models/cv/object_detection/hrnet/ixrt)      | 4.3.0     |
| PAA        | FP16  | [✅](models/cv/object_detection/paa/igie)        | [✅](models/cv/object_detection/paa/ixrt)        | 4.3.0     |
| RetinaFace | FP16  | [✅](models/cv/object_detection/retinaface/igie) | [✅](models/cv/object_detection/retinaface/ixrt) | 4.3.0     |
| RetinaNet  | FP16  | [✅](models/cv/object_detection/retinanet/igie)  | [✅](models/cv/object_detection/retinanet/ixrt)  | 4.3.0     |
| RTMDet     | FP16  | [✅](models/cv/object_detection/rtmdet/igie)     |                                                 | 4.3.0     |
| RTDETR     | FP16  | [✅](models/cv/object_detection/rtdetr/igie)     | [✅](models/cv/object_detection/rtdetr/ixrt)    | dev-only  |
|            | INT8  | [✅](models/cv/object_detection/rtdetr/igie)     |                                                 | dev-only   |
| SABL       | FP16  | [✅](models/cv/object_detection/sabl/igie)       |                                                 | 4.3.0     |
| SSD        | FP16  | [✅](models/cv/object_detection/ssd/igie)        |                                                 | 4.3.0     |
| YOLOF      | FP16  | [✅](models/cv/object_detection/yolof/igie)      | [✅](models/cv/object_detection/yolof/ixrt)    | 4.3.0     |
| YOLOv3     | FP16  | [✅](models/cv/object_detection/yolov3/igie)     | [✅](models/cv/object_detection/yolov3/ixrt)     | 4.3.0     |
|            | INT8  | [✅](models/cv/object_detection/yolov3/igie)     | [✅](models/cv/object_detection/yolov3/ixrt)     | 4.3.0     |
| YOLOv4     | FP16  | [✅](models/cv/object_detection/yolov4/igie)     | [✅](models/cv/object_detection/yolov4/ixrt)     | 4.3.0     |
|            | INT8  | [✅](models/cv/object_detection/yolov4/igie16)   | [✅](models/cv/object_detection/yolov4/ixrt16)   | 4.3.0     |
| YOLOv5m    | FP16  | [✅](models/cv/object_detection/yolov5m/igie)     | [✅](models/cv/object_detection/yolov5m/ixrt)     | 4.3.0     |
|            | INT8  | [✅](models/cv/object_detection/yolov5m/igie)     | [✅](models/cv/object_detection/yolov5m/ixrt)     | 4.3.0     |
| YOLOv5s    | FP16  | [✅](models/cv/object_detection/yolov5s/igie)    | [✅](models/cv/object_detection/yolov5s/ixrt)    | 4.3.0     |
|            | INT8  | [✅](models/cv/object_detection/yolov5s/igie)    | [✅](models/cv/object_detection/yolov5s/ixrt)    | 4.3.0     |
| YOLOv6s    | FP16  | [✅](models/cv/object_detection/yolov6s/igie)     | [✅](models/cv/object_detection/yolov6s/ixrt)     | 4.3.0     |
|            | INT8  |                                                 | [✅](models/cv/object_detection/yolov6s/ixrt)     | 4.3.0     |
| YOLOv7     | FP16  | [✅](models/cv/object_detection/yolov7/igie)     | [✅](models/cv/object_detection/yolov7/ixrt)     | 4.3.0     |
|            | INT8  | [✅](models/cv/object_detection/yolov7/igie)     | [✅](models/cv/object_detection/yolov7/ixrt)     | 4.3.0     |
| YOLOv8n    | FP16  | [✅](models/cv/object_detection/yolov8n/igie)    | [✅](models/cv/object_detection/yolov8n/ixrt)    | 4.3.0     |
|            | INT8  | [✅](models/cv/object_detection/yolov8n/igie)    | [✅](models/cv/object_detection/yolov8n/ixrt)    | 4.3.0     |
| YOLOv8s    | FP16  | [✅](models/cv/object_detection/yolov8s/igie)    |                                                  | 4.3.0     |
|            | INT8  | [✅](models/cv/object_detection/yolov8s/igie)    |                                                  | 4.3.0     |
| YOLOv9s    | FP16  | [✅](models/cv/object_detection/yolov9s/igie)     | [✅](models/cv/object_detection/yolov9s/ixrt)     | 4.3.0     |
|            | INT8  | [✅](models/cv/object_detection/yolov9s/igie)     |                                                   | 4.3.0     |
| YOLOv10s   | FP16  | [✅](models/cv/object_detection/yolov10s/igie)    | [✅](models/cv/object_detection/yolov10s/ixrt)    | 4.3.0     |
| YOLOv11m   | FP16  | [✅](models/cv/object_detection/yolov11m/igie)    |                                                   | 4.4.0     |
|            | INT8  | [✅](models/cv/object_detection/yolov11m/igie)     |                                                  | 4.4.0     |
| YOLOv11n   | FP16  | [✅](models/cv/object_detection/yolov11n/igie)    | [✅](models/cv/object_detection/yolov11n/ixrt)    | 4.3.0     |
|            | INT8  | [✅](models/cv/object_detection/yolov11n/igie)     |                                                  | 4.3.0     |
| YOLOv11s   | FP16  | [✅](models/cv/object_detection/yolov11s/igie)    |                                                   | 4.4.0     |
|            | INT8  | [✅](models/cv/object_detection/yolov11s/igie)     |                                                  | 4.4.0     |
| YOLOv12n   | FP16  | [✅](models/cv/object_detection/yolov12n/igie)    | [✅](models/cv/object_detection/yolov12n/ixrt)    | 4.3.0     |
|            | INT8  | [✅](models/cv/object_detection/yolov12n/igie)     |                                                   | 4.3.0     |
| YOLOv13n   | FP16  | [✅](models/cv/object_detection/yolov13n/igie)    | [✅](models/cv/object_detection/yolov13n/ixrt)    | 4.3.0     |
|            | INT8  | [✅](models/cv/object_detection/yolov13n/igie)     |                                                   | 4.3.0     |
| YOLOv26n   | FP16  | [✅](models/cv/object_detection/yolov26n/igie)    |                                                   | 4.4.0     |
| YOLOXm     | FP16  | [✅](models/cv/object_detection/yoloxm/igie)      | [✅](models/cv/object_detection/yoloxm/ixrt)      | 4.3.0     |
|            | INT8  | [✅](models/cv/object_detection/yoloxm/igie)      | [✅](models/cv/object_detection/yoloxm/ixrt)      | 4.3.0     |


| Model      | Prec. | PaddlePaddle                                            | IXUCA SDK |
|------------|-------|---------------------------------------------------------|-----------|
| RTDETR     | FP16  | [✅](models/cv/object_detection/rtdetr/paddlepaddle)    | dev-only  |

| Model      | Prec. | Pytorch                                            | IXUCA SDK |
|------------|-------|-------------------------------------------------|-----------|
| YOLOv8n       | FP16  | [✅](models/cv/object_detection/yolov8n/pytorch)   | dev-only    |

#### 人脸识别

| Model   | Prec. | IGIE | ixRT                                         | IXUCA SDK |
|---------|-------|------|----------------------------------------------|-----------|
| FaceNet | FP16  |      | [✅](models/cv/face_recognition/facenet/ixrt) | 4.3.0     |
|         | INT8  |      | [✅](models/cv/face_recognition/facenet/ixrt) | 4.3.0     |

#### 光学字符识别（OCR）

| Model         | Prec. | IGIE                                  |     ixRT                              | IXUCA SDK |
|---------------|-------|---------------------------------------|---------------------------------------|-----------|
| CRNN          | FP16  |                                         | [✅](models/cv/ocr/crnn/ixrt)      |  4.4.0     |
| DBNet         | FP16  |                                        |  [✅](models/cv/ocr/dbnet/ixrt)     |  4.4.0     |
| Kie_layoutXLM | FP16  | [✅](models/cv/ocr/kie_layoutxlm/igie) |                                     |  4.3.0     |
| SVTR          | FP16  | [✅](models/cv/ocr/svtr/igie)          |                                     |  4.3.0     |

#### 姿态估计

| Model                | Prec. | IGIE                                          | ixRT                                                     | IXUCA SDK |
|----------------------|-------|-----------------------------------------------|----------------------------------------------------------|-----------|
| HRNetPose            | FP16  | [✅](models/cv/pose_estimation/hrnetpose/igie) |                                                          | 4.3.0     |
| Lightweight OpenPose | FP16  |                                               | [✅](models/cv/pose_estimation/lightweight_openpose/ixrt) | 4.3.0     |
| RTMPose              | FP16  | [✅](models/cv/pose_estimation/rtmpose/igie)   | [✅](models/cv/pose_estimation/rtmpose/ixrt)              | 4.3.0     |

#### 实例分割

| Model      | Prec. | IGIE | ixRT                                                | IXUCA SDK |
|------------|-------|------|-----------------------------------------------------|-----------|
| Mask R-CNN | FP16  |      | [✅](models/cv/instance_segmentation/mask_rcnn/ixrt) | 4.2.0     |
| SOLOv1     | FP16  |      | [✅](models/cv/instance_segmentation/solov1/ixrt)    | 4.3.0     |

#### 语义分割

| Model | Prec. | IGIE                                           | ixRT                                           | IXUCA SDK |
|-------|-------|------------------------------------------------|------------------------------------------------|-----------|
| DDRNet | FP16  |                                               | [✅](models/cv/semantic_segmentation/ddrnet/ixrt)  | 4.4.0     |
| UNet  | FP16  | [✅](models/cv/semantic_segmentation/unet/igie) | [✅](models/cv/semantic_segmentation/unet/ixrt)  | 4.3.0     |

#### 多目标跟踪

| Model               | Prec. | IGIE                                               | ixRT | IXUCA SDK |
|---------------------|-------|----------------------------------------------------|------|-----------|
| FastReID            | FP16  | [✅](models/cv/multi_object_tracking/fastreid/igie) | [✅](models/cv/multi_object_tracking/fastreid/ixrt) | 4.3.0     |
| DeepSort            | FP16  | [✅](models/cv/multi_object_tracking/deepsort/igie) | [✅](models/cv/multi_object_tracking/deepsort/ixrt) | 4.3.0     |
|                     | INT8  | [✅](models/cv/multi_object_tracking/deepsort/igie) |      | 4.3.0     |
| RepNet-Vehicle-ReID | FP16  | [✅](models/cv/multi_object_tracking/repnet/igie)   | [✅](models/cv/multi_object_tracking/repnet/ixrt)   | 4.3.0     |

### 多模态

| Model               | Engine   |    Supported                                                           | IXUCA SDK |
|---------------------|----------|------------------------------------------------------------------------|-----------|
| Aria                |   vLLM   | [✅](models/multimodal/vision_language_model/aria/vllm)                | 4.3.0     |
| Chameleon-7B        |   vLLM   | [✅](models/multimodal/vision_language_model/chameleon_7b/vllm)        | 4.3.0     |
| CLIP                | IxFormer | [✅](models/multimodal/vision_language_model/clip/ixformer)            | 4.3.0     |
| DeepSeek-VL2-tiny   |   vLLM   | [✅](models/multimodal/vision_language_model/deepseek-vl2/vllm)        | 4.4.0     |
| Fuyu-8B             |   vLLM   | [✅](models/multimodal/vision_language_model/fuyu_8b/vllm)             | 4.3.0     |
| FLUX.1-Dev          | xDiT   | [✅](models/multimodal/diffusion_model/flux.1-dev/xdit)                 | 4.4.0     |
| H2OVL Mississippi   |   vLLM   | [✅](models/multimodal/vision_language_model/h2vol/vllm)               | 4.3.0     |
| HunyuanVideo        | xDiT   | [✅](models/multimodal/diffusion_model/hunyuan_video/xdit)              | 4.4.0     |
| HunyuanDiT-v1.2     | xDiT   | [✅](models/multimodal/diffusion_model/hunyuanDit-v1.2/xdit)           | 4.4.0     |
| Idefics3            |   vLLM   | [✅](models/multimodal/vision_language_model/idefics3/vllm)            | 4.3.0     |
| InternVL2-4B        |   vLLM   | [✅](models/multimodal/vision_language_model/intern_vl/vllm)           | 4.3.0     |
| LLaVA               |   vLLM   | [✅](models/multimodal/vision_language_model/llava/vllm)               | 4.3.0     |
| LLaVA-Next-Video-7B |   vLLM   | [✅](models/multimodal/vision_language_model/llava_next_video_7b/vllm) | 4.3.0     |
| Llama-3.2           |   vLLM   | [✅](models/multimodal/vision_language_model/llama-3.2/vllm)           | 4.3.0     |
| Pixtral             |   vLLM   | [✅](models/multimodal/vision_language_model/pixtral/vllm)             | 4.3.0     |
| Qwen-Image          |   ComfyUI  | [✅](models/multimodal/vision_language_model/qwen-image/comfyui)     | 4.4.0     |
| Stable Diffusion 1.5 | Diffusers   | [✅](models/multimodal/diffusion_model/stable-diffusion-1.5/diffusers)  | 4.3.0     |
| Stable Diffusion 2.1 | ixRT   | [✅](models/multimodal/diffusion_model/stable-diffusion-2.1/ixrt)  | 4.4.0     |
| Stable Diffusion 3 |   Diffusers   | [✅](models/multimodal/diffusion_model/stable-diffusion-3/diffusers)    | dev-only  |
| SD3-Medium          | xDiT   | [✅](models/multimodal/diffusion_model/stable-diffusion-3-medium/xdit) | 4.4.0     |
| Wan2.1-T2V-14B      | xDiT   | [✅](models/multimodal/diffusion_model/wan2.1-t2v-14b/xdit)            | 4.4.0     |
| Wan2.2-TI2V-5B      | xDiT   | [✅](models/multimodal/diffusion_model/wan2.2-ti2v-5b/xdit)            | 4.4.0     |
| Z-Image      | Diffusers   | [✅](models/multimodal/diffusion_model/z-image/diffusers)            | 4.4.0     |

### 自然语言处理（NLP）

#### 预训练语言模型（PLM）

| Model            | Prec. | IGIE                                      | ixRT                                      | IXUCA SDK |
|------------------|-------|-------------------------------------------|-------------------------------------------|-----------|
| ALBERT           | FP16  |                                           | [✅](models/nlp/plm/albert/ixrt)           | 4.3.0     |
| BERT Base NER    | INT8  | [✅](models/nlp/plm/bert_base_ner/igie)    |                                           | 4.3.0     |
| BERT Base SQuAD  | FP16  | [✅](models/nlp/plm/bert_base_squad/igie)  | [✅](models/nlp/plm/bert_base_squad/ixrt)  | 4.3.0     |
|                  | INT8  |                                           | [✅](models/nlp/plm/bert_base_squad/ixrt)  | 4.3.0     |
| BERT Large SQuAD | FP16  | [✅](models/nlp/plm/bert_large_squad/igie) | [✅](models/nlp/plm/bert_large_squad/ixrt) | 4.3.0     |
|                  | INT8  | [✅](models/nlp/plm/bert_large_squad/igie) | [✅](models/nlp/plm/bert_large_squad/ixrt) | 4.3.0     |
| DeBERTa          | FP16  |                                           | [✅](models/nlp/plm/deberta/ixrt)          | 4.3.0     |
| RoBERTa          | FP16  |                                           | [✅](models/nlp/plm/roberta/ixrt)          | 4.3.0     |
| RoFormer         | FP16  |                                           | [✅](models/nlp/plm/roformer/ixrt)         | 4.3.0     |
| VideoBERT        | FP16  |                                           | [✅](models/nlp/plm/videobert/ixrt)        | 4.2.0     |

### 语音

#### 语音识别

| Model           | Prec. | IGIE                                                | ixRT                                                      | IXUCA SDK |
|-----------------|-------|-----------------------------------------------------|-----------------------------------------------------------|-----------|
| Conformer       | FP16  | [✅](models/audio/speech_recognition/conformer/igie) | [✅](models/audio/speech_recognition/conformer/ixrt)       | 4.3.0     |
| DeepSpeech2     | FP16  |                                                      | [✅](models/speech/speech_recognition/deepspeech2/ixrt)     | 4.4.0     |
| Transformer ASR | FP16  |                                                     | [✅](models/audio/speech_recognition/transformer_asr/ixrt) | 4.2.0     |

### 其他

#### 推荐系统

| Model       | Prec. | IGIE | ixRT                                                 | IXUCA SDK |
|-------------|-------|------|------------------------------------------------------|-----------|
| Wide & Deep | FP16  |      | [✅](models/others/recommendation/wide_and_deep/ixrt) | 4.3.0     |

---

## 容器

| Docker Installer | IXUCA SDK | Introduction |
|-----------|--------|--------------|
| corex-docker-installer-4.3.0-*-py3.10-x86_64.run | 4.3.0 | 适用小模型推理     |
| corex-docker-installer-4.3.0-*-llm-py3.10-x86_64.run | 4.3.0 | 适用大模型推理  |

## 社区

### 治理

请参见 DeepSpark Code of Conduct on [Gitee](https://gitee.com/deep-spark/deepspark/blob/master/CODE_OF_CONDUCT.md) or on [GitHub](https://github.com/Deep-Spark/deepspark/blob/main/CODE_OF_CONDUCT.md)。

### 交流

请联系 <contact@deepspark.org.cn>。

### 贡献

请参见 [DeepSparkInference Contributing Guidelines](CONTRIBUTING.md)。

### 免责声明

DeepSparkInference 仅提供公共数据集的下载和预处理脚本。这些数据集不属于 DeepSparkInference，DeepSparkInference 也不对其质量或维护负责。请确保您具有这些数据集的使用许可，基于这些数据集训练的模型仅可用于非商业研究和教育。

致数据集所有者：

如果不希望您的数据集公布在 DeepSparkInference 上或希望更新 DeepSparkInference 中属于您的数据集，请在 Gitee 或 Github 上提交 issue，我们将按您的 issue 删除或更新。衷心感谢您对我们社区的支持和贡献。

## 许可证

本项目许可证遵循 [Apache-2.0](LICENSE)。
