# DeepSparkInference

DeepSparkInference推理模型库作为DeepSpark开源社区的核心项目，于2024年3月正式开源，一期甄选了48个推理模型示例，涵盖计算机视觉，自然语言处理，语音识别等领域，后续将逐步拓展更多AI领域。

DeepSparkInference中的模型提供了在国产推理引擎IGIE或IxRT下运行的推理示例和指导文档，部分模型提供了基于国产通用GPU[智铠100](https://www.iluvatar.com/productDetails?fullCode=cpjs-yj-tlxltt-zk100)的评测结果。

IGIE（Iluvatar GPU Inference Engine）是基于TVM框架研发的高性能、高通用、全流程的AI推理引擎。支持多框架模型导入、量化、图优化、多算子库支持、多后端支持、算子自动调优等特性，为推理场景提供易部署、高吞吐量、低延迟的完整方案。

IxRT（Iluvatar CoreX RunTime）是天数智芯自研的高性能推理引擎，专注于最大限度发挥天数智芯通用GPU 的性能，实现各领域模型的高性能推理。IxRT支持动态形状推理、插件和INT8/FP16推理等特性。

DeepSparkInference将按季度进行版本更新，后续会逐步丰富模型类别并拓展大模型推理。

## LLM (Large Language Model)

| Model                         | vLLM                                                   | TRT-LLM                               | TGI                                |
|-------------------------------|--------------------------------------------------------|---------------------------------------|------------------------------------|
| Baichuan2-7B                  | [✅](models/nlp/llm/baichuan2-7b/vllm)                  |                                       |                                    |
| ChatGLM-3-6B                  | [✅](models/nlp/llm/chatglm3-6b/vllm)                   |                                       |                                    |
| ChatGLM-3-6B-32K              | [✅](models/nlp/llm/chatglm3-6b-32k/vllm)               |                                       |                                    |
| DeepSeek-R1-Distill-Llama-8B  | [✅](models/nlp/llm/deepseek-r1-distill-llama-8b/vllm)  |                                       |                                    |
| DeepSeek-R1-Distill-Llama-70B | [✅](models/nlp/llm/deepseek-r1-distill-llama-70b/vllm) |                                       |                                    |
| DeepSeek-R1-Distill-Qwen-1.5B | [✅](models/nlp/llm/deepseek-r1-distill-qwen-1.5b/vllm) |                                       |                                    |
| DeepSeek-R1-Distill-Qwen-7B   | [✅](models/nlp/llm/deepseek-r1-distill-qwen-7b/vllm)   |                                       |                                    |
| DeepSeek-R1-Distill-Qwen-14B  | [✅](models/nlp/llm/deepseek-r1-distill-qwen-14b/vllm)  |                                       |                                    |
| DeepSeek-R1-Distill-Qwen-32B  | [✅](models/nlp/llm/deepseek-r1-distill-qwen-32b/vllm)  |                                       |                                    |
| Llama2-7B                     | [✅](models/nlp/llm/llama2-7b/vllm)                     | [✅](models/nlp/llm/llama2-7b/trtllm)  |                                    |
| Llama2-13B                    |                                                        | [✅](models/nlp/llm/llama2-13b/trtllm) |                                    |
| Llama2-70B                    |                                                        | [✅](models/nlp/llm/llama2-70b/trtllm) |                                    |
| Llama3-70B                    | [✅](models/nlp/llm/llama3-70b/vllm)                    |                                       |                                    |
| Qwen-7B                       | [✅](models/nlp/llm/qwen-7b/vllm)                       |                                       |                                    |
| Qwen1.5-7B                    | [✅](models/nlp/llm/qwen1.5-7b/vllm)                    |                                       | [✅](models/nlp/llm/qwen1.5-7b/tgi) |
| Qwen1.5-14B                   | [✅](models/nlp/llm/qwen1.5-14b/vllm)                   |                                       |                                    |
| Qwen1.5-32B Chat              | [✅](models/nlp/llm/qwen1.5-32b/vllm)                   |                                       |                                    |
| Qwen1.5-72B                   | [✅](models/nlp/llm/qwen1.5-72b/vllm)                   |                                       |                                    |
| Qwen2-7B Instruct             | [✅](models/nlp/llm/qwen2-7b/vllm)                      |                                       |                                    |
| Qwen2-72B Instruct            | [✅](models/nlp/llm/qwen2-72b/vllm)                     |                                       |                                    |
| StableLM2-1.6B                | [✅](models/nlp/llm/stablelm/vllm)                      |                                       |                                    |

## Computer Vision

### Classification

| Model                  | Prec. | IGIE                                                   | IxRT                                                      |
|------------------------|-------|--------------------------------------------------------|-----------------------------------------------------------|
| AlexNet                | FP16  | [✅](models/cv/classification/alexnet/igie)             | [✅](models/cv/classification/alexnet/ixrt)                |
|                        | INT8  | [✅](models/cv/classification/alexnet/igie)             | [✅](models/cv/classification/alexnet/ixrt)                |
| CLIP                   | FP16  | [✅](models/cv/classification/clip/igie)                |                                                           |
| Conformer-B            | FP16  | [✅](models/cv/classification/conformer_base/igie)      |                                                           |
| ConvNeXt-Base          | FP16  | [✅](models/cv/classification/convnext_base/igie)       | [✅](models/cv/classification/convnext_base/ixrt)          |
| ConvNext-S             | FP16  | [✅](models/cv/classification/convnext_s/igie)          |                                                           |
| ConvNeXt-Small         | FP16  | [✅](models/cv/classification/convnext_small/igie)      | [✅](models/cv/classification/convnext_small/ixrt)         |
| CSPDarkNet53           | FP16  | [✅](models/cv/classification/cspdarknet53/igie)        | [✅](models/cv/classification/cspdarknet53/ixrt)           |
|                        | INT8  |                                                        | [✅](models/cv/classification/cspdarknet53/ixrt)           |
| CSPResNet50            | FP16  | [✅](models/cv/classification/cspresnet50/igie)         | [✅](models/cv/classification/cspresnet50/ixrt)            |
|                        | INT8  |                                                        | [✅](models/cv/classification/cspresnet50/ixrt)            |
| DeiT-tiny              | FP16  | [✅](models/cv/classification/deit_tiny/igie)           | [✅](models/cv/classification/deit_tiny/ixrt)              |
| DenseNet121            | FP16  | [✅](models/cv/classification/densenet121/igie)         | [✅](models/cv/classification/densenet121/ixrt)            |
| DenseNet161            | FP16  | [✅](models/cv/classification/densenet161/igie)         | [✅](models/cv/classification/densenet161/ixrt)            |
| DenseNet169            | FP16  | [✅](models/cv/classification/densenet169/igie)         | [✅](models/cv/classification/densenet169/ixrt)            |
| DenseNet201            | FP16  | [✅](models/cv/classification/densenet201/igie)         | [✅](models/cv/classification/densenet201/ixrt)            |
| EfficientNet-B0        | FP16  | [✅](models/cv/classification/efficientnet_b0/igie)     | [✅](models/cv/classification/efficientnet_b0/ixrt)        |
|                        | INT8  |                                                        | [✅](models/cv/classification/efficientnet_b0/ixrt)        |
| EfficientNet-B1        | FP16  | [✅](models/cv/classification/efficientnet_b1/igie)     | [✅](models/cv/classification/efficientnet_b1/ixrt)        |
|                        | INT8  |                                                        | [✅](models/cv/classification/efficientnet_b1/ixrt)        |
| EfficientNet-B2        | FP16  | [✅](models/cv/classification/efficientnet_b2/igie)     | [✅](models/cv/classification/efficientnet_b2/ixrt)        |
| EfficientNet-B3        | FP16  | [✅](models/cv/classification/efficientnet_b3/igie)     | [✅](models/cv/classification/efficientnet_b3/ixrt)        |
| EfficientNet-B4        | FP16  | [✅](models/cv/classification/efficientnet_b4/igie)     |                                                           |
| EfficientNetV2         | FP16  | [✅](models/cv/classification/efficientnet_v2/igie)     | [✅](models/cv/classification/efficientnet_v2/ixrt)        |
|                        | INT8  |                                                        | [✅](models/cv/classification/efficientnet_v2/ixrt)        |
| EfficientNetv2_rw_t    | FP16  | [✅](models/cv/classification/efficientnetv2_rw_t/igie) | [✅](models/cv/classification/efficientnetv2_rw_t/ixrt)    |
| EfficientNetv2_s       | FP16  | [✅](models/cv/classification/efficientnet_v2_s/igie)   | [✅](models/cv/classification/efficientnet_v2_s/ixrt)      |
| GoogLeNet              | FP16  | [✅](models/cv/classification/googlenet/igie)           | [✅](models/cv/classification/googlenet/ixrt)              |
|                        | INT8  | [✅](models/cv/classification/googlenet/igie)           | [✅](models/cv/classification/googlenet/ixrt)              |
| HRNet-W18              | FP16  | [✅](models/cv/classification/hrnet_w18/igie)           | [✅](models/cv/classification/hrnet_w18/ixrt)              |
|                        | INT8  |                                                        | [✅](models/cv/classification/hrnet_w18/ixrt)              |
| InceptionV3            | FP16  | [✅](models/cv/classification/inception_v3/igie)        | [✅](models/cv/classification/inception_v3/ixrt)           |
|                        | INT8  | [✅](models/cv/classification/inception_v3/igie)        | [✅](models/cv/classification/inception_v3/ixrt)           |
| Inception-ResNet-V2    | FP16  |                                                        | [✅](models/cv/classification/inception_resnet_v2/ixrt)    |
|                        | INT8  |                                                        | [✅](models/cv/classification/inception_resnet_v2/ixrt)    |
| Mixer_B                | FP16  | [✅](models/cv/classification/mlp_mixer_base/igie)      |                                                           |
| MNASNet0_5             | FP16  | [✅](models/cv/classification/mnasnet0_5/igie)          |                                                           |
| MNASNet0_75            | FP16  | [✅](models/cv/classification/mnasnet0_75/igie)         |                                                           |
| MobileNetV2            | FP16  | [✅](models/cv/classification/mobilenet_v2/igie)        | [✅](models/cv/classification/mobilenet_v2/ixrt)           |
|                        | INT8  | [✅](models/cv/classification/mobilenet_v2/igie)        | [✅](models/cv/classification/mobilenet_v2/ixrt)           |
| MobileNetV3_Large      | FP16  | [✅](models/cv/classification/mobilenet_v3_large/igie)  |                                                           |
| MobileNetV3_Small      | FP16  | [✅](models/cv/classification/mobilenet_v3/igie)        | [✅](models/cv/classification/mobilenet_v3/ixrt)           |
| MViTv2_base            | FP16  | [✅](models/cv/classification/mvitv2_base/igie)         |                                                           |
| RegNet_x_16gf          | FP16  | [✅](models/cv/classification/regnet_x_16gf/igie)       |                                                           |
| RegNet_x_1_6gf         | FP16  | [✅](models/cv/classification/regnet_x_1_6gf/igie)      |                                                           |
| RegNet_y_1_6gf         | FP16  | [✅](models/cv/classification/regnet_y_1_6gf/igie)      |                                                           |
| RepVGG                 | FP16  | [✅](models/cv/classification/repvgg/igie)              | [✅](models/cv/classification/repvgg/ixrt)                 |
| Res2Net50              | FP16  | [✅](models/cv/classification/res2net50/igie)           | [✅](models/cv/classification/res2net50/ixrt)              |
|                        | INT8  |                                                        | [✅](models/cv/classification/res2net50/ixrt)              |
| ResNeSt50              | FP16  | [✅](models/cv/classification/resnest50/igie)           |                                                           |
| ResNet101              | FP16  | [✅](models/cv/classification/resnet101/igie)           | [✅](models/cv/classification/resnet101/ixrt)              |
|                        | INT8  | [✅](models/cv/classification/resnet101/igie)           | [✅](models/cv/classification/resnet101/ixrt)              |
| ResNet152              | FP16  | [✅](models/cv/classification/resnet152/igie)           |                                                           |
|                        | INT8  | [✅](models/cv/classification/resnet152/igie)           |                                                           |
| ResNet18               | FP16  | [✅](models/cv/classification/resnet18/igie)            | [✅](models/cv/classification/resnet18/ixrt)               |
|                        | INT8  | [✅](models/cv/classification/resnet18/igie)            | [✅](models/cv/classification/resnet18/ixrt)               |
| ResNet34               | FP16  |                                                        | [✅](models/cv/classification/resnet34/ixrt)               |
|                        | INT8  |                                                        | [✅](models/cv/classification/resnet34/ixrt)               |
| ResNet50               | FP16  | [✅](models/cv/classification/resnet50/igie)            | [✅](models/cv/classification/resnet50/ixrt)               |
|                        | INT8  | [✅](models/cv/classification/resnet50/igie)            |                                                           |
| ResNetV1D50            | FP16  | [✅](models/cv/classification/resnetv1d50/igie)         | [✅](models/cv/classification/resnetv1d50/ixrt)            |
|                        | INT8  |                                                        | [✅](models/cv/classification/resnetv1d50/ixrt)            |
| ResNeXt50_32x4d        | FP16  | [✅](models/cv/classification/resnext50_32x4d/igie)     | [✅](models/cv/classification/resnext50_32x4d/ixrt)        |
| ResNeXt101_64x4d       | FP16  | [✅](models/cv/classification/resnext101_64x4d/igie)    |                                                           |
| ResNeXt101_32x8d       | FP16  | [✅](models/cv/classification/resnext101_32x8d/igie)    |                                                           |
| SEResNet50             | FP16  | [✅](models/cv/classification/se_resnet50/igie)         |                                                           |
| ShuffleNetV1           | FP16  |                                                        | [✅](models/cv/classification/shufflenet_v1/ixrt)          |
| ShuffleNetV2_x0_5      | FP16  | [✅](models/cv/classification/shufflenetv2_x0_5/igie)   |                                                           |
| ShuffleNetV2_x1_0      | FP16  | [✅](models/cv/classification/shufflenetv2_x1_0/igie)   |                                                           |
| ShuffleNetV2_x1_5      | FP16  | [✅](models/cv/classification/shufflenetv2_x1_5/igie)   |                                                           |
| ShuffleNetV2_x2_0      | FP16  | [✅](models/cv/classification/shufflenetv2_x2_0/igie)   |                                                           |
| SqueezeNet 1.0         | FP16  | [✅](models/cv/classification/squeezenet_v1_0/igie)     | [✅](models/cv/classification/squeezenet_v1_0/ixrt)        |
|                        | INT8  |                                                        | [✅](models/cv/classification/squeezenet_v1_0/ixrt)        |
| SqueezeNet 1.1         | FP16  |                                                        | [✅](models/cv/classification/squeezenet_v1_1/ixrt)        |
|                        | INT8  |                                                        | [✅](models/cv/classification/squeezenet_v1_1/ixrt)        |
| SVT Base               | FP16  | [✅](models/cv/classification/svt_base/igie)            |                                                           |
| Swin Transformer       | FP16  | [✅](models/cv/classification/swin_transformer/igie)    |                                                           |
| Swin Transformer Large | FP16  |                                                        | [✅](models/cv/classification/swin_transformer_large/ixrt) |
| VGG11                  | FP16  | [✅](models/cv/classification/vgg11/igie)               |                                                           |
| VGG16                  | FP16  | [✅](models/cv/classification/vgg16/igie)               | [✅](models/cv/classification/vgg16/ixrt)                  |
|                        | INT8  | [✅](models/cv/classification/vgg16/igie)               |                                                           |
| Wide ResNet50          | FP16  | [✅](models/cv/classification/wide_resnet50/igie)       | [✅](models/cv/classification/wide_resnet50/ixrt)          |
|                        | INT8  | [✅](models/cv/classification/wide_resnet50/igie)       | [✅](models/cv/classification/wide_resnet50/ixrt)          |
| Wide ResNet101         | FP16  | [✅](models/cv/classification/wide_resnet101/igie)      |                                                           |

### Object Detection

| Model      | Prec. | IGIE                                            | IxRT                                            |
|------------|-------|-------------------------------------------------|-------------------------------------------------|
| ATSS       | FP16  | [✅](models/cv/object_detection/atss/igie)       |                                                 |
| CenterNet  | FP16  | [✅](models/cv/object_detection/centernet/igie)  | [✅](models/cv/object_detection/centernet/ixrt)  |
| DETR       | FP16  |                                                 | [✅](models/cv/object_detection/detr/ixrt)       |
| FCOS       | FP16  | [✅](models/cv/object_detection/fcos/igie)       | [✅](models/cv/object_detection/fcos/ixrt)       |
| FoveaBox   | FP16  | [✅](models/cv/object_detection/foveabox/igie)   | [✅](models/cv/object_detection/foveabox/ixrt)   |
| FSAF       | FP16  | [✅](models/cv/object_detection/fsaf/igie)       | [✅](models/cv/object_detection/fsaf/ixrt)       |
| HRNet      | FP16  | [✅](models/cv/object_detection/hrnet/igie)      | [✅](models/cv/object_detection/hrnet/ixrt)      |
| PAA        | FP16  | [✅](models/cv/object_detection/paa/igie)        |                                                 |
| RetinaFace | FP16  | [✅](models/cv/object_detection/retinaface/igie) | [✅](models/cv/object_detection/retinaface/ixrt) |
| RetinaNet  | FP16  | [✅](models/cv/object_detection/retinanet/igie)  |                                                 |
| RTMDet     | FP16  | [✅](models/cv/object_detection/rtmdet/igie)     |                                                 |
| SABL       | FP16  | [✅](models/cv/object_detection/sabl/igie)       |                                                 |
| YOLOv3     | FP16  | [✅](models/cv/object_detection/yolov3/igie)     | [✅](models/cv/object_detection/yolov3/ixrt)     |
|            | INT8  | [✅](models/cv/object_detection/yolov3/igie)     | [✅](models/cv/object_detection/yolov3/ixrt)     |
| YOLOv4     | FP16  | [✅](models/cv/object_detection/yolov4/igie)     | [✅](models/cv/object_detection/yolov4/ixrt)     |
|            | INT8  | [✅](models/cv/object_detection/yolov4/igie16)   | [✅](models/cv/object_detection/yolov4/ixrt16)   |
| YOLOv5     | FP16  | [✅](models/cv/object_detection/yolov5/igie)     | [✅](models/cv/object_detection/yolov5/ixrt)     |
|            | INT8  | [✅](models/cv/object_detection/yolov5/igie)     | [✅](models/cv/object_detection/yolov5/ixrt)     |
| YOLOv5s    | FP16  |                                                 | [✅](models/cv/object_detection/yolov5s/ixrt)    |
|            | INT8  |                                                 | [✅](models/cv/object_detection/yolov5s/ixrt)    |
| YOLOv6     | FP16  | [✅](models/cv/object_detection/yolov6/igie)     | [✅](models/cv/object_detection/yolov6/ixrt)     |
|            | INT8  |                                                 | [✅](models/cv/object_detection/yolov6/ixrt)     |
| YOLOv7     | FP16  | [✅](models/cv/object_detection/yolov7/igie)     | [✅](models/cv/object_detection/yolov7/ixrt)     |
|            | INT8  | [✅](models/cv/object_detection/yolov7/igie)     | [✅](models/cv/object_detection/yolov7/ixrt)     |
| YOLOv8     | FP16  | [✅](models/cv/object_detection/yolov8/igie)     | [✅](models/cv/object_detection/yolov8/ixrt)     |
|            | INT8  | [✅](models/cv/object_detection/yolov8/igie)     | [✅](models/cv/object_detection/yolov8/ixrt)     |
| YOLOv9     | FP16  | [✅](models/cv/object_detection/yolov9/igie)     |                                                 |
| YOLOv10    | FP16  | [✅](models/cv/object_detection/yolov10/igie)    |                                                 |
| YOLOv11    | FP16  | [✅](models/cv/object_detection/yolov11/igie)    |                                                 |
| YOLOX      | FP16  | [✅](models/cv/object_detection/yolox/igie)      | [✅](models/cv/object_detection/yolox/ixrt)      |
|            | INT8  | [✅](models/cv/object_detection/yolox/igie)      | [✅](models/cv/object_detection/yolox/ixrt)      |

### Face Recognition

| Model   | Prec. | IGIE | IxRT                             |
|---------|-------|------|----------------------------------|
| FaceNet | FP16  |      | [✅](models/cv/face_recognition/facenet/ixrt) |
|         | INT8  |      | [✅](models/cv/face_recognition/facenet/ixrt) |

### OCR (Optical Character Recognition)

| Model         | Prec. | IGIE                                  |
|---------------|-------|---------------------------------------|
| Kie_layoutXLM | FP16  | [✅](models/cv/ocr/kie_layoutxlm/igie) |
| SVTR          | FP16  | [✅](models/cv/ocr/svtr/igie)          |

### Pose Estimation

| Model                | Prec. | IGIE                                          | IxRT                                                     |
|----------------------|-------|-----------------------------------------------|----------------------------------------------------------|
| HRNetPose            | FP16  | [✅](models/cv/pose_estimation/hrnetpose/igie) |                                                          |
| Lightweight OpenPose | FP16  |                                               | [✅](models/cv/pose_estimation/lightweight_openpose/ixrt) |
| RTMPose              | FP16  | [✅](models/cv/pose_estimation/rtmpose/igie)   | [✅](models/cv/pose_estimation/rtmpose/ixrt)              |

### Instance Segmentation

| Model      | Prec. | IGIE | IxRT                                                |
|------------|-------|------|-----------------------------------------------------|
| Mask R-CNN | FP16  |      | [✅](models/cv/instance_segmentation/mask_rcnn/ixrt) |
| SOLOv1     | FP16  |      | [✅](models/cv/instance_segmentation/solov1/ixrt)    |

### Multi-Object Tracking

| Model               | Prec. | IGIE                                               | IxRT |
|---------------------|-------|----------------------------------------------------|------|
| FastReID            | FP16  | [✅](models/cv/multi_object_tracking/fastreid/igie) |      |
| DeepSort            | FP16  | [✅](models/cv/multi_object_tracking/deepsort/igie) |      |
|                     | INT8  | [✅](models/cv/multi_object_tracking/deepsort/igie) |      |
| RepNet-Vehicle-ReID | FP16  | [✅](models/cv/multi_object_tracking/repnet/igie)   |      |

## Multimodal

| Model               | vLLM                                                                  | IxFormer                                                   |
|---------------------|-----------------------------------------------------------------------|------------------------------------------------------------|
| Chameleon-7B        | [✅](models/multimodal/vision_language_model/chameleon_7b/vllm)        |                                                            |
| CLIP                |                                                                       | [✅](models/multimodal/vision_language_model/clip/ixformer) |
| Fuyu-8B             | [✅](models/multimodal/vision_language_model/fuyu_8b/vllm)             |                                                            |
| InternVL2-4B        | [✅](models/multimodal/vision_language_model/intern_vl/vllm)           |                                                            |
| LLaVA               | [✅](models/multimodal/vision_language_model/llava/vllm)               |                                                            |
| LLaVA-Next-Video-7B | [✅](models/multimodal/vision_language_model/llava_next_video_7b/vllm) |                                                            |
| MiniCPM V2          | [✅](models/multimodal/vision_language_model/minicpm_v_2/vllm)         |                                                            |

## NLP

### PLM (Pre-trained Language Model)

| Model            | Prec. | IGIE                                      | IxRT                                      |
|------------------|-------|-------------------------------------------|-------------------------------------------|
| ALBERT           | FP16  |                                           | [✅](models/nlp/plm/albert/ixrt)           |
| BERT Base NER    | INT8  | [✅](models/nlp/plm/bert_base_ner/igie)    |                                           |
| BERT Base SQuAD  | FP16  | [✅](models/nlp/plm/bert_base_squad/igie)  | [✅](models/nlp/plm/bert_base_squad/ixrt)  |
|                  | INT8  |                                           | [✅](models/nlp/plm/bert_base_squad/ixrt)  |
| BERT Large SQuAD | FP16  | [✅](models/nlp/plm/bert_large_squad/igie) | [✅](models/nlp/plm/bert_large_squad/ixrt) |
|                  | INT8  | [✅](models/nlp/plm/bert_large_squad/igie) | [✅](models/nlp/plm/bert_large_squad/ixrt) |
| DeBERTa          | FP16  |                                           | [✅](models/nlp/plm/deberta/ixrt)          |
| RoBERTa          | FP16  |                                           | [✅](models/nlp/plm/roberta/ixrt)          |
| RoFormer         | FP16  |                                           | [✅](models/nlp/plm/roformer/ixrt)         |
| VideoBERT        | FP16  |                                           | [✅](models/nlp/plm/videobert/ixrt)        |

## Audio

### Speech Recognition

| Model           | Prec. | IGIE                                                | IxRT                                                      |
|-----------------|-------|-----------------------------------------------------|-----------------------------------------------------------|
| Conformer       | FP16  | [✅](models/audio/speech_recognition/conformer/igie) | [✅](models/audio/speech_recognition/conformer/ixrt)       |
| Transformer ASR | FP16  |                                                     | [✅](models/audio/speech_recognition/transformer_asr/ixrt) |

## Others

### Recommendation Systems

| Model       | Prec. | IGIE | IxRT                                                 |
|-------------|-------|------|------------------------------------------------------|
| Wide & Deep | FP16  |      | [✅](models/others/recommendation/wide_and_deep/ixrt) |

---

## 社区

### 治理

请参见 DeepSpark Code of Conduct on [Gitee](https://gitee.com/deep-spark/deepspark/blob/master/CODE_OF_CONDUCT.md) or on [GitHub](https://github.com/Deep-Spark/deepspark/blob/main/CODE_OF_CONDUCT.md)。

### 交流

请联系 <contact@deepspark.org.cn>。

### 贡献

请参见 [DeepSparkInference Contributing Guidelines](CONTRIBUTING.md)。

### 免责声明

DeepSparkInference仅提供公共数据集的下载和预处理脚本。这些数据集不属于DeepSparkInference，DeepSparkInference也不对其质量或维护负责。请确保您具有这些数据集的使用许可，基于这些数据集训练的模型仅可用于非商业研究和教育。

致数据集所有者：

如果不希望您的数据集公布在DeepSparkInference上或希望更新DeepSparkInference中属于您的数据集，请在Gitee或Github上提交issue，我们将按您的issue删除或更新。衷心感谢您对我们社区的支持和贡献。

## 许可证

本项目许可证遵循[Apache-2.0](LICENSE)。
