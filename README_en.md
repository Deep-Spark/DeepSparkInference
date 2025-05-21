[<img src="https://img.shields.io/badge/Language-English-blue.svg">](README_en.md) [<img src="https://img.shields.io/badge/语言-简体中文-red.svg">](README.md)

# DeepSparkInference

<div align="center" style="line-height: 1;">
  <a href="https://www.deepspark.org.cn"><img alt="Homepage"
    src="https://img.shields.io/badge/DeepSpark-Homepage-blue.svg"/></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-dfd.svg"></a>
  <a href="https://gitee.com/deep-spark/deepsparkinference/releases/latest"><img src="https://img.shields.io/github/v/release/deep-spark/deepsparkinference?color=ffa"></a>
</div>
<br>

DeepSparkInference ModelZoo, as a core project of the DeepSpark open-source community, was officially open-sourced in
March 2024. The first release selected 48 inference model examples, covering fields such as computer vision, natural
language processing, and speech recognition. More AI domains will be gradually expanded in the future.

The models in DeepSparkInference provide inference examples and guidance documents for running on inference engines IGIE
or IxRT self-developed by Iluvatar CoreX. Some models provide evaluation results based on the self-developed GPGPU
Zhikai 100.

IGIE (Iluvatar GPU Inference Engine) is a high-performance, highly gene, and end-to-end AI inference engine developed
based on the TVM framework. It supports multi-framework model, quantization, graph optimization, multi-operator library
support, multi-backend support, and automatic operator tuning, providing an easy-to-deploy, high-throughput, and
low-latency complete solution for inference scenarios.

IxRT (Iluvatar CoreX RunTime) is a high-performance inference engine independently developed by Iluvatar CoreX, focusing
on maximizing the performance of Iluvatar CoreX's GPGPU and achieving high-performance inference for models in various
fields. IxRT supports features such as dynamic shape inference, plugins, and INT8/FP16 inference.

DeepSparkInference will be updated quarterly, and model categories will be gradually enriched, with large model
inference to be expanded in the future.

## ModelZoo

### LLM (Large Language Model)

| Model                         | vLLM                                                   | TRT-LLM                               | TGI                                | IXUCA SDK |
|-------------------------------|--------------------------------------------------------|---------------------------------------|------------------------------------|-----------|
| Baichuan2-7B                  | [✅](models/nlp/llm/baichuan2-7b/vllm)                  |                                       |                                    | 4.2.0     |
| ChatGLM-3-6B                  | [✅](models/nlp/llm/chatglm3-6b/vllm)                   |                                       |                                    | 4.2.0     |
| ChatGLM-3-6B-32K              | [✅](models/nlp/llm/chatglm3-6b-32k/vllm)               |                                       |                                    | 4.2.0     |
| DeepSeek-R1-Distill-Llama-8B  | [✅](models/nlp/llm/deepseek-r1-distill-llama-8b/vllm)  |                                       |                                    | 4.2.0     |
| DeepSeek-R1-Distill-Llama-70B | [✅](models/nlp/llm/deepseek-r1-distill-llama-70b/vllm) |                                       |                                    | 4.2.0     |
| DeepSeek-R1-Distill-Qwen-1.5B | [✅](models/nlp/llm/deepseek-r1-distill-qwen-1.5b/vllm) |                                       |                                    | 4.2.0     |
| DeepSeek-R1-Distill-Qwen-7B   | [✅](models/nlp/llm/deepseek-r1-distill-qwen-7b/vllm)   |                                       |                                    | 4.2.0     |
| DeepSeek-R1-Distill-Qwen-14B  | [✅](models/nlp/llm/deepseek-r1-distill-qwen-14b/vllm)  |                                       |                                    | 4.2.0     |
| DeepSeek-R1-Distill-Qwen-32B  | [✅](models/nlp/llm/deepseek-r1-distill-qwen-32b/vllm)  |                                       |                                    | 4.2.0     |
| Llama2-7B                     | [✅](models/nlp/llm/llama2-7b/vllm)                     | [✅](models/nlp/llm/llama2-7b/trtllm)  |                                    | 4.2.0     |
| Llama2-13B                    |                                                        | [✅](models/nlp/llm/llama2-13b/trtllm) |                                    | 4.2.0     |
| Llama2-70B                    |                                                        | [✅](models/nlp/llm/llama2-70b/trtllm) |                                    | 4.2.0     |
| Llama3-70B                    | [✅](models/nlp/llm/llama3-70b/vllm)                    |                                       |                                    | 4.2.0     |
| Qwen-7B                       | [✅](models/nlp/llm/qwen-7b/vllm)                       |                                       |                                    | 4.2.0     |
| Qwen1.5-7B                    | [✅](models/nlp/llm/qwen1.5-7b/vllm)                    |                                       | [✅](models/nlp/llm/qwen1.5-7b/tgi) | 4.2.0     |
| Qwen1.5-14B                   | [✅](models/nlp/llm/qwen1.5-14b/vllm)                   |                                       |                                    | 4.2.0     |
| Qwen1.5-32B Chat              | [✅](models/nlp/llm/qwen1.5-32b/vllm)                   |                                       |                                    | 4.2.0     |
| Qwen1.5-72B                   | [✅](models/nlp/llm/qwen1.5-72b/vllm)                   |                                       |                                    | 4.2.0     |
| Qwen2-7B Instruct             | [✅](models/nlp/llm/qwen2-7b/vllm)                      |                                       |                                    | 4.2.0     |
| Qwen2-72B Instruct            | [✅](models/nlp/llm/qwen2-72b/vllm)                     |                                       |                                    | 4.2.0     |
| StableLM2-1.6B                | [✅](models/nlp/llm/stablelm/vllm)                      |                                       |                                    | 4.2.0     |

### Computer Vision

#### Classification

| Model                  | Prec. | IGIE                                                   | IxRT                                                      | IXUCA SDK |
|------------------------|-------|--------------------------------------------------------|-----------------------------------------------------------|-----------|
| AlexNet                | FP16  | [✅](models/cv/classification/alexnet/igie)             | [✅](models/cv/classification/alexnet/ixrt)                | 4.2.0     |
|                        | INT8  | [✅](models/cv/classification/alexnet/igie)             | [✅](models/cv/classification/alexnet/ixrt)                | 4.2.0     |
| CLIP                   | FP16  | [✅](models/cv/classification/clip/igie)                |                                                           | 4.2.0     |
| Conformer-B            | FP16  | [✅](models/cv/classification/conformer_base/igie)      |                                                           | 4.2.0     |
| ConvNeXt-Base          | FP16  | [✅](models/cv/classification/convnext_base/igie)       | [✅](models/cv/classification/convnext_base/ixrt)          | 4.2.0     |
| ConvNext-S             | FP16  | [✅](models/cv/classification/convnext_s/igie)          |                                                           | 4.2.0     |
| ConvNeXt-Small         | FP16  | [✅](models/cv/classification/convnext_small/igie)      | [✅](models/cv/classification/convnext_small/ixrt)         | 4.2.0     |
| ConvNeXt-Tiny          | FP16  | [✅](models/cv/classification/convnext_tiny/igie)       |                                                           | 4.2.0     |
| CSPDarkNet53           | FP16  | [✅](models/cv/classification/cspdarknet53/igie)        | [✅](models/cv/classification/cspdarknet53/ixrt)           | 4.2.0     |
|                        | INT8  |                                                        | [✅](models/cv/classification/cspdarknet53/ixrt)           | 4.2.0     |
| CSPResNet50            | FP16  | [✅](models/cv/classification/cspresnet50/igie)         | [✅](models/cv/classification/cspresnet50/ixrt)            | 4.2.0     |
|                        | INT8  |                                                        | [✅](models/cv/classification/cspresnet50/ixrt)            | 4.2.0     |
| CSPResNeXt50           | FP16  | [✅](models/cv/classification/cspresnext50/igie)        |                                                           | 4.2.0     |
| DeiT-tiny              | FP16  | [✅](models/cv/classification/deit_tiny/igie)           | [✅](models/cv/classification/deit_tiny/ixrt)              | 4.2.0     |
| DenseNet121            | FP16  | [✅](models/cv/classification/densenet121/igie)         | [✅](models/cv/classification/densenet121/ixrt)            | 4.2.0     |
| DenseNet161            | FP16  | [✅](models/cv/classification/densenet161/igie)         | [✅](models/cv/classification/densenet161/ixrt)            | 4.2.0     |
| DenseNet169            | FP16  | [✅](models/cv/classification/densenet169/igie)         | [✅](models/cv/classification/densenet169/ixrt)            | 4.2.0     |
| DenseNet201            | FP16  | [✅](models/cv/classification/densenet201/igie)         | [✅](models/cv/classification/densenet201/ixrt)            | 4.2.0     |
| EfficientNet-B0        | FP16  | [✅](models/cv/classification/efficientnet_b0/igie)     | [✅](models/cv/classification/efficientnet_b0/ixrt)        | 4.2.0     |
|                        | INT8  |                                                        | [✅](models/cv/classification/efficientnet_b0/ixrt)        | 4.2.0     |
| EfficientNet-B1        | FP16  | [✅](models/cv/classification/efficientnet_b1/igie)     | [✅](models/cv/classification/efficientnet_b1/ixrt)        | 4.2.0     |
|                        | INT8  |                                                        | [✅](models/cv/classification/efficientnet_b1/ixrt)        | 4.2.0     |
| EfficientNet-B2        | FP16  | [✅](models/cv/classification/efficientnet_b2/igie)     | [✅](models/cv/classification/efficientnet_b2/ixrt)        | 4.2.0     |
| EfficientNet-B3        | FP16  | [✅](models/cv/classification/efficientnet_b3/igie)     | [✅](models/cv/classification/efficientnet_b3/ixrt)        | 4.2.0     |
| EfficientNet-B4        | FP16  | [✅](models/cv/classification/efficientnet_b4/igie)     |                                                           | 4.2.0     |
| EfficientNet-B5        | FP16  | [✅](models/cv/classification/efficientnet_b5/igie)     |                                                           | 4.2.0     |
| EfficientNetV2         | FP16  | [✅](models/cv/classification/efficientnet_v2/igie)     | [✅](models/cv/classification/efficientnet_v2/ixrt)        | 4.2.0     |
|                        | INT8  |                                                        | [✅](models/cv/classification/efficientnet_v2/ixrt)        | 4.2.0     |
| EfficientNetv2_rw_t    | FP16  | [✅](models/cv/classification/efficientnetv2_rw_t/igie) | [✅](models/cv/classification/efficientnetv2_rw_t/ixrt)    | 4.2.0     |
| EfficientNetv2_s       | FP16  | [✅](models/cv/classification/efficientnet_v2_s/igie)   | [✅](models/cv/classification/efficientnet_v2_s/ixrt)      | 4.2.0     |
| GoogLeNet              | FP16  | [✅](models/cv/classification/googlenet/igie)           | [✅](models/cv/classification/googlenet/ixrt)              | 4.2.0     |
|                        | INT8  | [✅](models/cv/classification/googlenet/igie)           | [✅](models/cv/classification/googlenet/ixrt)              | 4.2.0     |
| HRNet-W18              | FP16  | [✅](models/cv/classification/hrnet_w18/igie)           | [✅](models/cv/classification/hrnet_w18/ixrt)              | 4.2.0     |
|                        | INT8  |                                                        | [✅](models/cv/classification/hrnet_w18/ixrt)              | 4.2.0     |
| InceptionV3            | FP16  | [✅](models/cv/classification/inception_v3/igie)        | [✅](models/cv/classification/inception_v3/ixrt)           | 4.2.0     |
|                        | INT8  | [✅](models/cv/classification/inception_v3/igie)        | [✅](models/cv/classification/inception_v3/ixrt)           | 4.2.0     |
| Inception-ResNet-V2    | FP16  |                                                        | [✅](models/cv/classification/inception_resnet_v2/ixrt)    | 4.2.0     |
|                        | INT8  |                                                        | [✅](models/cv/classification/inception_resnet_v2/ixrt)    | 4.2.0     |
| Mixer_B                | FP16  | [✅](models/cv/classification/mlp_mixer_base/igie)      |                                                           | 4.2.0     |
| MNASNet0_5             | FP16  | [✅](models/cv/classification/mnasnet0_5/igie)          |                                                           | 4.2.0     |
| MNASNet0_75            | FP16  | [✅](models/cv/classification/mnasnet0_75/igie)         |                                                           | 4.2.0     |
| MNASNet1_0             | FP16  | [✅](models/cv/classification/mnasnet1_0/igie)          |                                                           | 4.2.0     |
| MobileNetV2            | FP16  | [✅](models/cv/classification/mobilenet_v2/igie)        | [✅](models/cv/classification/mobilenet_v2/ixrt)           | 4.2.0     |
|                        | INT8  | [✅](models/cv/classification/mobilenet_v2/igie)        | [✅](models/cv/classification/mobilenet_v2/ixrt)           | 4.2.0     |
| MobileNetV3_Large      | FP16  | [✅](models/cv/classification/mobilenet_v3_large/igie)  |                                                           | 4.2.0     |
| MobileNetV3_Small      | FP16  | [✅](models/cv/classification/mobilenet_v3/igie)        | [✅](models/cv/classification/mobilenet_v3/ixrt)           | 4.2.0     |
| MViTv2_base            | FP16  | [✅](models/cv/classification/mvitv2_base/igie)         |                                                           | 4.2.0     |
| RegNet_x_16gf          | FP16  | [✅](models/cv/classification/regnet_x_16gf/igie)       |                                                           | 4.2.0     |
| RegNet_x_1_6gf         | FP16  | [✅](models/cv/classification/regnet_x_1_6gf/igie)      |                                                           | 4.2.0     |
| RegNet_x_3_2gf         | FP16  | [✅](models/cv/classification/regnet_x_3_2gf/igie)      |                                                           | 4.2.0     |
| RegNet_y_1_6gf         | FP16  | [✅](models/cv/classification/regnet_y_1_6gf/igie)      |                                                           | 4.2.0     |
| RegNet_y_16gf          | FP16  | [✅](models/cv/classification/regnet_y_16gf/igie)       |                                                           | 4.2.0     |
| RepVGG                 | FP16  | [✅](models/cv/classification/repvgg/igie)              | [✅](models/cv/classification/repvgg/ixrt)                 | 4.2.0     |
| Res2Net50              | FP16  | [✅](models/cv/classification/res2net50/igie)           | [✅](models/cv/classification/res2net50/ixrt)              | 4.2.0     |
|                        | INT8  |                                                        | [✅](models/cv/classification/res2net50/ixrt)              | 4.2.0     |
| ResNeSt50              | FP16  | [✅](models/cv/classification/resnest50/igie)           |                                                           | 4.2.0     |
| ResNet101              | FP16  | [✅](models/cv/classification/resnet101/igie)           | [✅](models/cv/classification/resnet101/ixrt)              | 4.2.0     |
|                        | INT8  | [✅](models/cv/classification/resnet101/igie)           | [✅](models/cv/classification/resnet101/ixrt)              | 4.2.0     |
| ResNet152              | FP16  | [✅](models/cv/classification/resnet152/igie)           |                                                           | 4.2.0     |
|                        | INT8  | [✅](models/cv/classification/resnet152/igie)           |                                                           | 4.2.0     |
| ResNet18               | FP16  | [✅](models/cv/classification/resnet18/igie)            | [✅](models/cv/classification/resnet18/ixrt)               | 4.2.0     |
|                        | INT8  | [✅](models/cv/classification/resnet18/igie)            | [✅](models/cv/classification/resnet18/ixrt)               | 4.2.0     |
| ResNet34               | FP16  |                                                        | [✅](models/cv/classification/resnet34/ixrt)               | 4.2.0     |
|                        | INT8  |                                                        | [✅](models/cv/classification/resnet34/ixrt)               | 4.2.0     |
| ResNet50               | FP16  | [✅](models/cv/classification/resnet50/igie)            | [✅](models/cv/classification/resnet50/ixrt)               | 4.2.0     |
|                        | INT8  | [✅](models/cv/classification/resnet50/igie)            |                                                           | 4.2.0     |
| ResNetV1D50            | FP16  | [✅](models/cv/classification/resnetv1d50/igie)         | [✅](models/cv/classification/resnetv1d50/ixrt)            | 4.2.0     |
|                        | INT8  |                                                        | [✅](models/cv/classification/resnetv1d50/ixrt)            | 4.2.0     |
| ResNeXt50_32x4d        | FP16  | [✅](models/cv/classification/resnext50_32x4d/igie)     | [✅](models/cv/classification/resnext50_32x4d/ixrt)        | 4.2.0     |
| ResNeXt101_64x4d       | FP16  | [✅](models/cv/classification/resnext101_64x4d/igie)    |  [✅](models/cv/classification/resnext101_64x4d/ixrt)                                                         | 4.2.0     |
| ResNeXt101_32x8d       | FP16  | [✅](models/cv/classification/resnext101_32x8d/igie)    |  [✅](models/cv/classification/resnext101_32x8d/ixrt)                                                         | 4.2.0     |
| SEResNet50             | FP16  | [✅](models/cv/classification/se_resnet50/igie)         |                                                           | 4.2.0     |
| ShuffleNetV1           | FP16  |                                                        | [✅](models/cv/classification/shufflenet_v1/ixrt)          | 4.2.0     |
| ShuffleNetV2_x0_5      | FP16  | [✅](models/cv/classification/shufflenetv2_x0_5/igie)   | [✅](models/cv/classification/shufflenetv2_x0_5/ixrt)      | 4.2.0     |
| ShuffleNetV2_x1_0      | FP16  | [✅](models/cv/classification/shufflenetv2_x1_0/igie)   | [✅](models/cv/classification/shufflenetv2_x1_0/ixrt)      | 4.2.0     |
| ShuffleNetV2_x1_5      | FP16  | [✅](models/cv/classification/shufflenetv2_x1_5/igie)   | [✅](models/cv/classification/shufflenetv2_x1_5/ixrt)      | 4.2.0     |
| ShuffleNetV2_x2_0      | FP16  | [✅](models/cv/classification/shufflenetv2_x2_0/igie)   | [✅](models/cv/classification/shufflenetv2_x2_0/ixrt)      | 4.2.0     |
| SqueezeNet 1.0         | FP16  | [✅](models/cv/classification/squeezenet_v1_0/igie)     | [✅](models/cv/classification/squeezenet_v1_0/ixrt)        | 4.2.0     |
|                        | INT8  |                                                        | [✅](models/cv/classification/squeezenet_v1_0/ixrt)        | 4.2.0     |
| SqueezeNet 1.1         | FP16  | [✅](models/cv/classification/squeezenet_v1_1/igie)     | [✅](models/cv/classification/squeezenet_v1_1/ixrt)        | 4.2.0     |
|                        | INT8  |                                                        | [✅](models/cv/classification/squeezenet_v1_1/ixrt)        | 4.2.0     |
| SVT Base               | FP16  | [✅](models/cv/classification/svt_base/igie)            |                                                           | 4.2.0     |
| Swin Transformer       | FP16  | [✅](models/cv/classification/swin_transformer/igie)    |                                                           | 4.2.0     |
| Swin Transformer Large | FP16  |                                                        | [✅](models/cv/classification/swin_transformer_large/ixrt) | 4.2.0     |
| Twins_PCPVT            | FP16  | [✅](models/cv/classification/twins_pcpvt/igie)         |                                                           | 4.2.0     |
| VAN_B0                 | FP16  | [✅](models/cv/classification/van_b0/igie)              |                                                           | 4.2.0     |
| VGG11                  | FP16  | [✅](models/cv/classification/vgg11/igie)               |                                                           | 4.2.0     |
| VGG16                  | FP16  | [✅](models/cv/classification/vgg16/igie)               | [✅](models/cv/classification/vgg16/ixrt)                  | 4.2.0     |
|                        | INT8  | [✅](models/cv/classification/vgg16/igie)               |                                                           | 4.2.0     |
| VGG19                  | FP16  | [✅](models/cv/classification/vgg19/igie)               |                                                           | 4.2.0     |
| VGG19_BN               | FP16  | [✅](models/cv/classification/vgg19_bn/igie)            |                                                           | 4.2.0     |
| ViT                    | FP16  | [✅](models/cv/classification/vit/igie)                 |                                                           | 4.2.0     |
| Wide ResNet50          | FP16  | [✅](models/cv/classification/wide_resnet50/igie)       | [✅](models/cv/classification/wide_resnet50/ixrt)          | 4.2.0     |
|                        | INT8  | [✅](models/cv/classification/wide_resnet50/igie)       | [✅](models/cv/classification/wide_resnet50/ixrt)          | 4.2.0     |
| Wide ResNet101         | FP16  | [✅](models/cv/classification/wide_resnet101/igie)      |                                                           | 4.2.0     |

#### Object Detection

| Model      | Prec. | IGIE                                            | IxRT                                            | IXUCA SDK |
|------------|-------|-------------------------------------------------|-------------------------------------------------|-----------|
| ATSS       | FP16  | [✅](models/cv/object_detection/atss/igie)       |                                                 | 4.2.0     |
| CenterNet  | FP16  | [✅](models/cv/object_detection/centernet/igie)  | [✅](models/cv/object_detection/centernet/ixrt)  | 4.2.0     |
| DETR       | FP16  |                                                 | [✅](models/cv/object_detection/detr/ixrt)       | 4.2.0     |
| FCOS       | FP16  | [✅](models/cv/object_detection/fcos/igie)       | [✅](models/cv/object_detection/fcos/ixrt)       | 4.2.0     |
| FoveaBox   | FP16  | [✅](models/cv/object_detection/foveabox/igie)   | [✅](models/cv/object_detection/foveabox/ixrt)   | 4.2.0     |
| FSAF       | FP16  | [✅](models/cv/object_detection/fsaf/igie)       | [✅](models/cv/object_detection/fsaf/ixrt)       | 4.2.0     |
| GFL        | FP16  | [✅](models/cv/object_detection/gfl/igie)        |                                                 | 4.2.0     |
| HRNet      | FP16  | [✅](models/cv/object_detection/hrnet/igie)      | [✅](models/cv/object_detection/hrnet/ixrt)      | 4.2.0     |
| PAA        | FP16  | [✅](models/cv/object_detection/paa/igie)        |                                                 | 4.2.0     |
| RetinaFace | FP16  | [✅](models/cv/object_detection/retinaface/igie) | [✅](models/cv/object_detection/retinaface/ixrt) | 4.2.0     |
| RetinaNet  | FP16  | [✅](models/cv/object_detection/retinanet/igie)  |                                                 | 4.2.0     |
| RTMDet     | FP16  | [✅](models/cv/object_detection/rtmdet/igie)     |                                                 | 4.2.0     |
| SABL       | FP16  | [✅](models/cv/object_detection/sabl/igie)       |                                                 | 4.2.0     |
| YOLOv3     | FP16  | [✅](models/cv/object_detection/yolov3/igie)     | [✅](models/cv/object_detection/yolov3/ixrt)     | 4.2.0     |
|            | INT8  | [✅](models/cv/object_detection/yolov3/igie)     | [✅](models/cv/object_detection/yolov3/ixrt)     | 4.2.0     |
| YOLOv4     | FP16  | [✅](models/cv/object_detection/yolov4/igie)     | [✅](models/cv/object_detection/yolov4/ixrt)     | 4.2.0     |
|            | INT8  | [✅](models/cv/object_detection/yolov4/igie16)   | [✅](models/cv/object_detection/yolov4/ixrt16)   | 4.2.0     |
| YOLOv5     | FP16  | [✅](models/cv/object_detection/yolov5/igie)     | [✅](models/cv/object_detection/yolov5/ixrt)     | 4.2.0     |
|            | INT8  | [✅](models/cv/object_detection/yolov5/igie)     | [✅](models/cv/object_detection/yolov5/ixrt)     | 4.2.0     |
| YOLOv5s    | FP16  |                                                 | [✅](models/cv/object_detection/yolov5s/ixrt)    | 4.2.0     |
|            | INT8  |                                                 | [✅](models/cv/object_detection/yolov5s/ixrt)    | 4.2.0     |
| YOLOv6     | FP16  | [✅](models/cv/object_detection/yolov6/igie)     | [✅](models/cv/object_detection/yolov6/ixrt)     | 4.2.0     |
|            | INT8  |                                                 | [✅](models/cv/object_detection/yolov6/ixrt)     | 4.2.0     |
| YOLOv7     | FP16  | [✅](models/cv/object_detection/yolov7/igie)     | [✅](models/cv/object_detection/yolov7/ixrt)     | 4.2.0     |
|            | INT8  | [✅](models/cv/object_detection/yolov7/igie)     | [✅](models/cv/object_detection/yolov7/ixrt)     | 4.2.0     |
| YOLOv8     | FP16  | [✅](models/cv/object_detection/yolov8/igie)     | [✅](models/cv/object_detection/yolov8/ixrt)     | 4.2.0     |
|            | INT8  | [✅](models/cv/object_detection/yolov8/igie)     | [✅](models/cv/object_detection/yolov8/ixrt)     | 4.2.0     |
| YOLOv9     | FP16  | [✅](models/cv/object_detection/yolov9/igie)     | [✅](models/cv/object_detection/yolov9/ixrt)     | 4.2.0     |
| YOLOv10    | FP16  | [✅](models/cv/object_detection/yolov10/igie)    | [✅](models/cv/object_detection/yolov10/ixrt)    | 4.2.0     |
| YOLOv11    | FP16  | [✅](models/cv/object_detection/yolov11/igie)    | [✅](models/cv/object_detection/yolov11/ixrt)    | 4.2.0     |
| YOLOv12    | FP16  | [✅](models/cv/object_detection/yolov12/igie)    |                                                 | 4.2.0     |
| YOLOX      | FP16  | [✅](models/cv/object_detection/yolox/igie)      | [✅](models/cv/object_detection/yolox/ixrt)      | 4.2.0     |
|            | INT8  | [✅](models/cv/object_detection/yolox/igie)      | [✅](models/cv/object_detection/yolox/ixrt)      | 4.2.0     |

#### Face Recognition

| Model   | Prec. | IGIE | IxRT                                         | IXUCA SDK |
|---------|-------|------|----------------------------------------------|-----------|
| FaceNet | FP16  |      | [✅](models/cv/face_recognition/facenet/ixrt) | 4.2.0     |
|         | INT8  |      | [✅](models/cv/face_recognition/facenet/ixrt) | 4.2.0     |

#### OCR (Optical Character Recognition)

| Model         | Prec. | IGIE                                  | IXUCA SDK |
|---------------|-------|---------------------------------------|-----------|
| Kie_layoutXLM | FP16  | [✅](models/cv/ocr/kie_layoutxlm/igie) | 4.2.0     |
| SVTR          | FP16  | [✅](models/cv/ocr/svtr/igie)          | 4.2.0     |

#### Pose Estimation

| Model                | Prec. | IGIE                                          | IxRT                                                     | IXUCA SDK |
|----------------------|-------|-----------------------------------------------|----------------------------------------------------------|-----------|
| HRNetPose            | FP16  | [✅](models/cv/pose_estimation/hrnetpose/igie) |                                                          | 4.2.0     |
| Lightweight OpenPose | FP16  |                                               | [✅](models/cv/pose_estimation/lightweight_openpose/ixrt) | 4.2.0     |
| RTMPose              | FP16  | [✅](models/cv/pose_estimation/rtmpose/igie)   | [✅](models/cv/pose_estimation/rtmpose/ixrt)              | 4.2.0     |

#### Instance Segmentation

| Model      | Prec. | IGIE | IxRT                                                | IXUCA SDK |
|------------|-------|------|-----------------------------------------------------|-----------|
| Mask R-CNN | FP16  |      | [✅](models/cv/instance_segmentation/mask_rcnn/ixrt) | 4.2.0     |
| SOLOv1     | FP16  |      | [✅](models/cv/instance_segmentation/solov1/ixrt)    | 4.2.0     |

#### Semantic Segmentation

| Model | Prec. | IGIE                                           | IxRT | IXUCA SDK |
|-------|-------|------------------------------------------------|------|-----------|
| UNet  | FP16  | [✅](models/cv/semantic_segmentation/unet/igie) |      | 4.2.0     |

#### Multi-Object Tracking

| Model               | Prec. | IGIE                                               | IxRT | IXUCA SDK |
|---------------------|-------|----------------------------------------------------|------|-----------|
| FastReID            | FP16  | [✅](models/cv/multi_object_tracking/fastreid/igie) |      | 4.2.0     |
| DeepSort            | FP16  | [✅](models/cv/multi_object_tracking/deepsort/igie) |      | 4.2.0     |
|                     | INT8  | [✅](models/cv/multi_object_tracking/deepsort/igie) |      | 4.2.0     |
| RepNet-Vehicle-ReID | FP16  | [✅](models/cv/multi_object_tracking/repnet/igie)   |      | 4.2.0     |

### Multimodal

| Model               | vLLM                                                                  | IxFormer                                                   | IXUCA SDK |
|---------------------|-----------------------------------------------------------------------|------------------------------------------------------------|-----------|
| Aria                | [✅](models/multimodal/vision_language_model/aria/vllm)                |                                                            | 4.2.0     |
| Chameleon-7B        | [✅](models/multimodal/vision_language_model/chameleon_7b/vllm)        |                                                            | 4.2.0     |
| CLIP                |                                                                       | [✅](models/multimodal/vision_language_model/clip/ixformer) | 4.2.0     |
| Fuyu-8B             | [✅](models/multimodal/vision_language_model/fuyu_8b/vllm)             |                                                            | 4.2.0     |
| H2OVL Mississippi   | [✅](models/multimodal/vision_language_model/h2vol/vllm)               |                                                            | 4.2.0     |
| Idefics3            | [✅](models/multimodal/vision_language_model/idefics3/vllm)            |                                                            | 4.2.0     |
| InternVL2-4B        | [✅](models/multimodal/vision_language_model/intern_vl/vllm)           |                                                            | 4.2.0     |
| LLaVA               | [✅](models/multimodal/vision_language_model/llava/vllm)               |                                                            | 4.2.0     |
| LLaVA-Next-Video-7B | [✅](models/multimodal/vision_language_model/llava_next_video_7b/vllm) |                                                            | 4.2.0     |
| Llama-3.2           | [✅](models/multimodal/vision_language_model/llama-3.2/vllm)              |                                                            | 4.2.0     |
| MiniCPM-V 2         | [✅](models/multimodal/vision_language_model/minicpm_v/vllm)           |                                                            | 4.2.0     |
| Pixtral             | [✅](models/multimodal/vision_language_model/pixtral/vllm)             |                                                            | 4.2.0     |

### NLP

#### PLM (Pre-trained Language Model)

| Model            | Prec. | IGIE                                      | IxRT                                      | IXUCA SDK |
|------------------|-------|-------------------------------------------|-------------------------------------------|-----------|
| ALBERT           | FP16  |                                           | [✅](models/nlp/plm/albert/ixrt)           | 4.2.0     |
| BERT Base NER    | INT8  | [✅](models/nlp/plm/bert_base_ner/igie)    |                                           | 4.2.0     |
| BERT Base SQuAD  | FP16  | [✅](models/nlp/plm/bert_base_squad/igie)  | [✅](models/nlp/plm/bert_base_squad/ixrt)  | 4.2.0     |
|                  | INT8  |                                           | [✅](models/nlp/plm/bert_base_squad/ixrt)  | 4.2.0     |
| BERT Large SQuAD | FP16  | [✅](models/nlp/plm/bert_large_squad/igie) | [✅](models/nlp/plm/bert_large_squad/ixrt) | 4.2.0     |
|                  | INT8  | [✅](models/nlp/plm/bert_large_squad/igie) | [✅](models/nlp/plm/bert_large_squad/ixrt) | 4.2.0     |
| DeBERTa          | FP16  |                                           | [✅](models/nlp/plm/deberta/ixrt)          | 4.2.0     |
| RoBERTa          | FP16  |                                           | [✅](models/nlp/plm/roberta/ixrt)          | 4.2.0     |
| RoFormer         | FP16  |                                           | [✅](models/nlp/plm/roformer/ixrt)         | 4.2.0     |
| VideoBERT        | FP16  |                                           | [✅](models/nlp/plm/videobert/ixrt)        | 4.2.0     |

### Audio

#### Speech Recognition

| Model           | Prec. | IGIE                                                | IxRT                                                      | IXUCA SDK |
|-----------------|-------|-----------------------------------------------------|-----------------------------------------------------------|-----------|
| Conformer       | FP16  | [✅](models/audio/speech_recognition/conformer/igie) | [✅](models/audio/speech_recognition/conformer/ixrt)       | 4.2.0     |
| Transformer ASR | FP16  |                                                     | [✅](models/audio/speech_recognition/transformer_asr/ixrt) | 4.2.0     |

### Others

#### Recommendation Systems

| Model       | Prec. | IGIE | IxRT                                                 | IXUCA SDK |
|-------------|-------|------|------------------------------------------------------|-----------|
| Wide & Deep | FP16  |      | [✅](models/others/recommendation/wide_and_deep/ixrt) | 4.2.0     |

---

## Community

### Code of Conduct

Please refer to DeepSpark Code of Conduct on
[Gitee](https://gitee.com/deep-spark/deepspark/blob/master/CODE_OF_CONDUCT.md) or on
[GitHub](https://github.com/Deep-Spark/deepspark/blob/main/CODE_OF_CONDUCT.md).

### Contact

Please contact <contact@deepspark.org.cn>.

### Contribution

Please refer to the [DeepSparkInference Contributing Guidelines](CONTRIBUTING.md).

### Disclaimers

DeepSparkInference only provides download and preprocessing scripts for public datasets. These datasets do not belong to
DeepSparkInference, and DeepSparkInference is not responsible for their quality or maintenance. Please ensure that you
have the necessary usage licenses for these datasets. Models trained based on these datasets can only be used for
non-commercial research and education purposes.

To dataset owners:

If you do not want your dataset to be published on DeepSparkInference or wish to update the dataset that belongs to you
on DeepSparkInference, please submit an issue on Gitee or Github. We will delete or update it according to your issue.
We sincerely appreciate your support and contributions to our community.

## License

This project is released under [Apache-2.0](LICENSE) License.
