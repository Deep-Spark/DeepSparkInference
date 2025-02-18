# DeepSparkInference

DeepSparkInference推理模型库作为DeepSpark开源社区的核心项目，于2024年3月正式开源，一期甄选了48个推理模型示例，涵盖计算机视觉，自然语言处理，语音识别等领域，后续将逐步拓展更多AI领域。

DeepSparkInference中的模型提供了在国产推理引擎IGIE或IxRT下运行的推理示例和指导文档，部分模型提供了基于国产通用GPU[智铠100](https://www.iluvatar.com/productDetails?fullCode=cpjs-yj-tlxltt-zk100)的评测结果。

IGIE（Iluvatar GPU Inference Engine）是基于TVM框架研发的高性能、高通用、全流程的AI推理引擎。支持多框架模型导入、量化、图优化、多算子库支持、多后端支持、算子自动调优等特性，为推理场景提供易部署、高吞吐量、低延迟的完整方案。

IxRT（Iluvatar CoreX RunTime）是天数智芯自研的高性能推理引擎，专注于最大限度发挥天数智芯通用GPU 的性能，实现各领域模型的高性能推理。IxRT支持动态形状推理、插件和INT8/FP16推理等特性。

DeepSparkInference将按季度进行版本更新，后续会逐步丰富模型类别并拓展大模型推理。

## Computer Vision

### Classification

| Models                 | Precision | IGIE                                                                  | IxRT                                                                  |
|------------------------|-----------|-----------------------------------------------------------------------|-----------------------------------------------------------------------|
| AlexNet                | FP16      | [✅](models/cv/classification/alexnet/igie/README.md#fp16)             | [✅](models/cv/classification/alexnet/ixrt/README.md#fp16)             |
|                        | INT8      | [✅](models/cv/classification/alexnet/igie/README.md#int8)             | [✅](models/cv/classification/alexnet/ixrt/README.md#int8)             |
| CLIP                   | FP16      | [✅](models/cv/classification/clip/igie/README.md#fp16)                | [✅](models/cv/classification/clip/ixformer/README.md#fp16)            |
| Conformer-B            | FP16      | [✅](models/cv/classification/conformer_base/igie/README.md#fp16)      |                                                                       |
| ConvNeXt-Base          | FP16      | [✅](models/cv/classification/convnext_base/igie/README.md#fp16)       | [✅](models/cv/classification/convnext_base/ixrt/README.md#fp16)       |
| ConvNext-S (OpenMMLab) | FP16      | [✅](models/cv/classification/convnext_s/igie/README.md#fp16)          |                                                                       |
| ConvNeXt-Small         | FP16      | [✅](models/cv/classification/convnext_small/igie/README.md#fp16)      | [✅](models/cv/classification/convnext_small/ixrt/README.md#fp16)      |
| CSPDarkNet53           | FP16      | [✅](models/cv/classification/cspdarknet53/igie/README.md#fp16)        | [✅](models/cv/classification/cspdarknet53/ixrt/README.md#fp16)        |
|                        | INT8      |                                                                       | [✅](models/cv/classification/cspdarknet53/ixrt/README.md#int8)        |
| CSPResNet50            | FP16      | [✅](models/cv/classification/cspresnet50/igie/README.md#fp16)         | [✅](models/cv/classification/cspresnet50/ixrt/README.md#fp16)         |
|                        | INT8      |                                                                       | [✅](models/cv/classification/cspresnet50/ixrt/README.md#int8)         |
| DeiT-tiny              | FP16      | [✅](models/cv/classification/deit_tiny/igie/README.md#fp16)           | [✅](models/cv/classification/deit_tiny/ixrt/README.md#fp16)           |
| DenseNet121            | FP16      | [✅](models/cv/classification/densenet121/igie/README.md#fp16)         | [✅](models/cv/classification/densenet121/ixrt/README.md#fp16)         |
| DenseNet161            | FP16      | [✅](models/cv/classification/densenet161/igie/README.md#fp16)         | [✅](models/cv/classification/densenet161/ixrt/README.md#fp16)         |
| DenseNet169            | FP16      | [✅](models/cv/classification/densenet169/igie/README.md#fp16)         | [✅](models/cv/classification/densenet169/ixrt/README.md#fp16)         |
| DenseNet201            | FP16      | [✅](models/cv/classification/densenet201/igie/README.md#fp16)         | [✅](models/cv/classification/densenet201/ixrt/README.md#fp16)         |
| EfficientNet-B0        | FP16      | [✅](models/cv/classification/efficientnet_b0/igie/README.md#fp16)     | [✅](models/cv/classification/efficientnet_b0/ixrt/README.md#fp16)     |
|                        | INT8      |                                                                       | [✅](models/cv/classification/efficientnet_b0/ixrt/README.md#int8)     |
| EfficientNet-B1        | FP16      | [✅](models/cv/classification/efficientnet_b1/igie/README.md#fp16)     | [✅](models/cv/classification/efficientnet_b1/ixrt/README.md#fp16)     |
|                        | INT8      |                                                                       | [✅](models/cv/classification/efficientnet_b1/ixrt/README.md#int8)     |
| EfficientNet-B2        | FP16      | [✅](models/cv/classification/efficientnet_b2/igie/README.md#fp16)     | [✅](models/cv/classification/efficientnet_b2/ixrt/README.md#fp16)     |
| EfficientNet-B3        | FP16      | [✅](models/cv/classification/efficientnet_b3/igie/README.md#fp16)     | [✅](models/cv/classification/efficientnet_b3/ixrt/README.md#fp16)     |
| EfficientNet-B4        | FP16      | [✅](models/cv/classification/efficientnet_b4/igie/README.md#fp16)     |                                                                       |
| EfficientNetV2         | FP16      | [✅](models/cv/classification/efficientnet_v2/igie/README.md#fp16)     | [✅](models/cv/classification/efficientnet_v2/ixrt/README.md#fp16)     |
|                        | INT8      |                                                                       | [✅](models/cv/classification/efficientnet_v2/ixrt/README.md#int8)     |
| EfficientNetv2_rw_t    | FP16      | [✅](models/cv/classification/efficientnetv2_rw_t/igie/README.md#fp16) | [✅](models/cv/classification/efficientnetv2_rw_t/ixrt/README.md#fp16) |
| EfficientNetv2_s       | FP16      | [✅](models/cv/classification/efficientnet_v2_s/igie/README.md#fp16)   | [✅](models/cv/classification/efficientnet_v2_s/ixrt/README.md#fp16)   |
| GoogLeNet              | FP16      | [✅](models/cv/classification/googlenet/igie/README.md#fp16)           | [✅](models/cv/classification/googlenet/ixrt/README.md#fp16)           |
|                        | INT8      | [✅](models/cv/classification/googlenet/igie/README.md#int8)           | [✅](models/cv/classification/googlenet/ixrt/README.md#int8)           |
| HRNet-W18              | FP16      | [✅](models/cv/classification/hrnet_w18/igie/README.md#fp16)           | [✅](models/cv/classification/hrnet_w18/ixrt/README.md#fp16)           |
|                        | INT8      |                                                                       | [✅](models/cv/classification/hrnet_w18/ixrt/README.md#int8)           |
| InceptionV3            | FP16      | [✅](models/cv/classification/inception_v3/igie/README.md#fp16)        | [✅](models/cv/classification/inception_v3/ixrt/README.md#fp16)        |
|                        | INT8      | [✅](models/cv/classification/inception_v3/igie/README.md#int8)        | [✅](models/cv/classification/inception_v3/ixrt/README.md#int8)        |
| Inception_ResNet_V2    | FP16      |                                                                       | [✅](models/cv/classification/inceptionresnetv2/ixrt/README.md#fp16)   |
|                        | INT8      |                                                                       | [✅](models/cv/classification/inceptionresnetv2/ixrt/README.md#int8)   |
| Mixer_B                | FP16      | [✅](models/cv/classification/mlp_mixer_base/igie/README.md#fp16)      |                                                                       |
| MNASNet0_5             | FP16      | [✅](models/cv/classification/mnasnet0_5/igie/README.md#fp16)          |                                                                       |
| MNASNet0_75            | FP16      | [✅](models/cv/classification/mnasnet0_75/igie/README.md#fp16)         |                                                                       |
| MobileNetV2            | FP16      | [✅](models/cv/classification/mobilenet_v2/igie/README.md#fp16)        | [✅](models/cv/classification/mobilenet_v2/ixrt/README.md#fp16)        |
|                        | INT8      | [✅](models/cv/classification/mobilenet_v2/igie/README.md#int8)        | [✅](models/cv/classification/mobilenet_v2/ixrt/README.md#int8)        |
| MobileNetV3_Large      | FP16      | [✅](models/cv/classification/mobilenet_v3_large/igie/README.md#fp16)  |                                                                       |
| MobileNetV3_Small      | FP16      | [✅](models/cv/classification/mobilenet_v3/igie/README.md#fp16)        | [✅](models/cv/classification/mobilenet_v3/ixrt/README.md#fp16)        |
| MViTv2_base            | FP16      | [✅](models/cv/classification/mvitv2_base/igie/README.md#fp16)         |                                                                       |
| RegNet_x_16gf          | FP16      | [✅](models/cv/classification/regnet_x_16gf/igie/README.md#fp16)       |                                                                       |
| RegNet_x_1_6gf         | FP16      | [✅](models/cv/classification/regnet_x_1_6gf/igie/README.md#fp16)      |                                                                       |
| RegNet_y_1_6gf         | FP16      | [✅](models/cv/classification/regnet_y_1_6gf/igie/README.md#fp16)      |                                                                       |
| RepVGG                 | FP16      | [✅](models/cv/classification/repvgg/igie/README.md#fp16)              | [✅](models/cv/classification/repvgg/ixrt/README.md#fp16)              |
| Res2Net50              | FP16      | [✅](models/cv/classification/res2net50/igie/README.md#fp16)           | [✅](models/cv/classification/res2net50/ixrt/README.md#fp16)           |
|                        | INT8      |                                                                       | [✅](models/cv/classification/res2net50/ixrt/README.md#int8)           |
| ResNeSt50              | FP16      | [✅](models/cv/classification/resnest50/igie/README.md#fp16)           |                                                                       |
| ResNet101              | FP16      | [✅](models/cv/classification/resnet101/igie/README.md#fp16)           | [✅](models/cv/classification/resnet101/ixrt/README.md#fp16)           |
|                        | INT8      | [✅](models/cv/classification/resnet101/igie/README.md#int8)           | [✅](models/cv/classification/resnet101/ixrt/README.md#int8)           |
| ResNet152              | FP16      | [✅](models/cv/classification/resnet152/igie/README.md#fp16)           |                                                                       |
|                        | INT8      | [✅](models/cv/classification/resnet152/igie/README.md#int8)           |                                                                       |
| ResNet18               | FP16      | [✅](models/cv/classification/resnet18/igie/README.md#fp16)            | [✅](models/cv/classification/resnet18/ixrt/README.md#fp16)            |
|                        | INT8      | [✅](models/cv/classification/resnet18/igie/README.md#int8)            | [✅](models/cv/classification/resnet18/ixrt/README.md#int8)            |
| ResNet34               | FP16      |                                                                       | [✅](models/cv/classification/resnet34/ixrt/README.md#fp16)            |
|                        | INT8      |                                                                       | [✅](models/cv/classification/resnet34/ixrt/README.md#int8)            |
| ResNet50               | FP16      | [✅](models/cv/classification/resnet50/igie/README.md#fp16)            | [✅](models/cv/classification/resnet50/ixrt/README.md#fp16)            |
|                        | INT8      | [✅](models/cv/classification/resnet50/igie/README.md#int8)            |                                                                       |
| ResNetV1D50            | FP16      | [✅](models/cv/classification/resnetv1d50/igie/README.md#fp16)         | [✅](models/cv/classification/resnetv1d50/ixrt/README.md#fp16)         |
|                        | INT8      |                                                                       | [✅](models/cv/classification/resnetv1d50/ixrt/README.md#int8)         |
| ResNeXt50_32x4d        | FP16      | [✅](models/cv/classification/resnext50_32x4d/igie/README.md#fp16)     | [✅](models/cv/classification/resnext50_32x4d/ixrt/README.md#fp16)     |
| ResNeXt101_64x4d       | FP16      | [✅](models/cv/classification/resnext101_64x4d/igie/README.md#fp16)    |                                                                       |
| ResNeXt101_32x8d       | FP16      | [✅](models/cv/classification/resnext101_32x8d/igie/README.md#fp16)    |                                                                       |
| SEResNet50             | FP16      | [✅](models/cv/classification/se_resnet50/igie/README.md#fp16)         |                                                                       |
| ShuffleNetV1           | FP16      |                                                                       | [✅](models/cv/classification/shufflenet_v1/ixrt/README.md#fp16)       |
| ShuffleNetV2_x0_5      | FP16      | [✅](models/cv/classification/shufflenetv2_x0_5/igie/README.md#fp16)   |                                                                       |
| ShuffleNetV2_x1_0      | FP16      | [✅](models/cv/classification/shufflenetv2_x1_0/igie/README.md#fp16)   |                                                                       |
| ShuffleNetV2_x1_5      | FP16      | [✅](models/cv/classification/shufflenetv2_x1_5/igie/README.md#fp16)   |                                                                       |
| ShuffleNetV2_x2_0      | FP16      | [✅](models/cv/classification/shufflenetv2_x2_0/igie/README.md#fp16)   |                                                                       |
| SqueezeNet 1.0         | FP16      | [✅](models/cv/classification/squeezenet_v1_0/igie/README.md#fp16)     | [✅](models/cv/classification/squeezenet_v1_0/ixrt/README.md#fp16)     |
|                        | INT8      |                                                                       | [✅](models/cv/classification/squeezenet_v1_0/ixrt/README.md#int8)     |
| SqueezeNet 1.1         | FP16      |                                                                       | [✅](models/cv/classification/squeezenet_v1_1/ixrt/README.md#fp16)     |
|                        | INT8      |                                                                       | [✅](models/cv/classification/squeezenet_v1_1/ixrt/README.md#int8)     |
| SVT Base               | FP16      | [✅](models/cv/classification/svt_base/igie/README.md#fp16)            |                                                                       |
| Swin Transformer       | FP16      | [✅](models/cv/classification/swin_transformer/igie/README.md#fp16)    |                                                                       |
| Swin Transformer Large | FP16      |                                                                       | [✅](models/cv/classification/swin_transformer_large/ixrt/README.md)   |
| VGG11                  | FP16      | [✅](models/cv/classification/vgg11/igie/README.md#fp16)               |                                                                       |
| VGG16                  | FP16      | [✅](models/cv/classification/vgg16/igie/README.md#fp16)               | [✅](models/cv/classification/vgg16/ixrt/README.md#fp16)               |
|                        | INT8      | [✅](models/cv/classification/vgg16/igie/README.md#int8)               |                                                                       |
| Wide ResNet50          | FP16      | [✅](models/cv/classification/wide_resnet50/igie/README.md#fp16)       | [✅](models/cv/classification/wide_resnet50/ixrt/README.md#fp16)       |
|                        | INT8      | [✅](models/cv/classification/wide_resnet50/igie/README.md#int8)       | [✅](models/cv/classification/wide_resnet50/ixrt/README.md#int8)       |
| Wide ResNet101         | FP16      | [✅](models/cv/classification/wide_resnet101/igie/README.md#fp16)      |                                                                       |

### Detection

| Models     | Precision | IGIE                                                    | IxRT                                                    |
|------------|-----------|---------------------------------------------------------|---------------------------------------------------------|
| ATSS       | FP16      | [✅](models/cv/detection/atss/igie/README.md#fp16)       |                                                         |
| CenterNet  | FP16      | [✅](models/cv/detection/centernet/igie/README.md#fp16)  | [✅](models/cv/detection/centernet/ixrt/README.md#fp16)  |
| DETR       | FP16      |                                                         | [✅](models/cv/detection/detr/ixrt/README.md#fp16)       |
| FCOS       | FP16      | [✅](models/cv/detection/fcos/igie/README.md#fp16)       | [✅](models/cv/detection/fcos/ixrt/README.md#fp16)       |
| FoveaBox   | FP16      | [✅](models/cv/detection/foveabox/igie/README.md#fp16)   | [✅](models/cv/detection/foveabox/ixrt/README.md#fp16)   |
| FSAF       | FP16      | [✅](models/cv/detection/fsaf/igie/README.md#fp16)       | [✅](models/cv/detection/fsaf/ixrt/README.md#fp16)       |
| HRNet      | FP16      | [✅](models/cv/detection/hrnet/igie/README.md#fp16)      | [✅](models/cv/detection/hrnet/ixrt/README.md#fp16)      |
| PAA        | FP16      | [✅](models/cv/detection/paa/igie/README.md#fp16)        |                                                         |
| RetinaFace | FP16      | [✅](models/cv/detection/retinaface/igie/README.md#fp16) | [✅](models/cv/detection/retinaface/ixrt/README.md#fp16) |
| RetinaNet  | FP16      | [✅](models/cv/detection/retinanet/igie/README.md#fp16)  |                                                         |
| RTMDet     | FP16      | [✅](models/cv/detection/rtmdet/igie/README.md#fp16)     |                                                         |
| SABL       | FP16      | [✅](models/cv/detection/sabl/igie/README.md#fp16)       |                                                         |
| YOLOv3     | FP16      | [✅](models/cv/detection/yolov3/igie/README.md#fp16)     | [✅](models/cv/detection/yolov3/ixrt/README.md#fp16)     |
|            | INT8      | [✅](models/cv/detection/yolov3/igie/README.md#int8)     | [✅](models/cv/detection/yolov3/ixrt/README.md#int8)     |
| YOLOv4     | FP16      | [✅](models/cv/detection/yolov4/igie/README.md#fp16)     | [✅](models/cv/detection/yolov4/ixrt/README.md#fp16)     |
|            | INT8      | [✅](models/cv/detection/yolov4/igie/README.md#int816)   | [✅](models/cv/detection/yolov4/ixrt/README.md#int816)   |
| YOLOv5     | FP16      | [✅](models/cv/detection/yolov5/igie/README.md#fp16)     | [✅](models/cv/detection/yolov5/ixrt/README.md#fp16)     |
|            | INT8      | [✅](models/cv/detection/yolov5/igie/README.md#int8)     | [✅](models/cv/detection/yolov5/ixrt/README.md#int8)     |
| YOLOv5s    | FP16      |                                                         | [✅](models/cv/detection/yolov5s/ixrt/README.md#fp16)    |
|            | INT8      |                                                         | [✅](models/cv/detection/yolov5s/ixrt/README.md#int8)    |
| YOLOv6     | FP16      | [✅](models/cv/detection/yolov6/igie/README.md#fp16)     | [✅](models/cv/detection/yolov6/ixrt/README.md#fp16)     |
|            | INT8      |                                                         | [✅](models/cv/detection/yolov6/ixrt/README.md#int8)     |
| YOLOv7     | FP16      | [✅](models/cv/detection/yolov7/igie/README.md#fp16)     | [✅](models/cv/detection/yolov7/ixrt/README.md#fp16)     |
|            | INT8      | [✅](models/cv/detection/yolov7/igie/README.md#int8)     | [✅](models/cv/detection/yolov7/ixrt/README.md#int8)     |
| YOLOv8     | FP16      | [✅](models/cv/detection/yolov8/igie/README.md#fp16)     | [✅](models/cv/detection/yolov8/ixrt/README.md#fp16)     |
|            | INT8      | [✅](models/cv/detection/yolov8/igie/README.md#int8)     | [✅](models/cv/detection/yolov8/ixrt/README.md#int8)     |
| YOLOv9     | FP16      | [✅](models/cv/detection/yolov9/igie/README.md#fp16)     |                                                         |
| YOLOv10    | FP16      | [✅](models/cv/detection/yolov10/igie/README.md#fp16)    |                                                         |
| YOLOv11    | FP16      | [✅](models/cv/detection/yolov11/igie/README.md#fp16)    |                                                         |
| YOLOX      | FP16      | [✅](models/cv/detection/yolox/igie/README.md#fp16)      | [✅](models/cv/detection/yolox/ixrt/README.md#fp16)      |
|            | INT8      | [✅](models/cv/detection/yolox/igie/README.md#int8)      | [✅](models/cv/detection/yolox/ixrt/README.md#int8)      |

### Face Recognition

| Models  | Precision | IGIE | IxRT                                            |
|---------|-----------|------|-------------------------------------------------|
| FaceNet | FP16      |      | [✅](models/cv/face/facenet/ixrt/README.md#fp16) |
|         | INT8      |      | [✅](models/cv/face/facenet/ixrt/README.md#int8) |

### OCR

| Models        | Precision | IGIE                                                 |
|---------------|-----------|------------------------------------------------------|
| Kie_layoutXLM | FP16      | [✅](models/cv/ocr/kie_layoutxlm/igie/README.md#fp16) |
| SVTR          | FP16      | [✅](models/cv/ocr/svtr/igie/README.md#fp16)          |

### Pose Estimation

| Models               | Precision | IGIE                                                         | IxRT                                                                    |
|----------------------|-----------|--------------------------------------------------------------|-------------------------------------------------------------------------|
| HRNetPose            | FP16      | [✅](models/cv/pose_estimation/hrnetpose/igie/README.md#fp16) |                                                                         |
| Lightweight OpenPose | FP16      |                                                              | [✅](models/cv/pose_estimation/lightweight_openpose/ixrt/README.md#fp16) |
| RTMPose              | FP16      | [✅](models/cv/pose_estimation/rtmpose/igie/README.md#fp16)   | [✅](models/cv/pose_estimation/rtmpose/ixrt/README.md#fp16)              |

### Recommendation Systems

| Models      | Precision | IGIE | IxRT                                                                   |
|-------------|-----------|------|------------------------------------------------------------------------|
| Wide & Deep | FP16      |      | [✅](models/recommendation/ctr-prediction/widedeep/ixrt/README.md#fp16) |

### Segmentation

| Models     | Precision | IGIE | IxRT                                                      |
|------------|-----------|------|-----------------------------------------------------------|
| Mask R-CNN | FP16      |      | [✅](models/cv/segmentation/mask_rcnn/ixrt/README.md#fp16) |
| SOLOv1     | FP16      |      | [✅](models/cv/segmentation/solov1/ixrt/README.md#fp16)    |

### Trace

| Models              | Precision | IGIE                                              | IxRT |
|---------------------|-----------|---------------------------------------------------|------|
| FastReID            | FP16      | [✅](models/cv/trace/fastreid/igie/README.md#fp16) |      |
| DeepSort            | FP16      | [✅](models/cv/trace/deepsort/igie/README.md#fp16) |      |
|                     | INT8      | [✅](models/cv/trace/deepsort/igie/README.md#int8) |      |
| RepNet-Vehicle-ReID | FP16      | [✅](models/cv/trace/repnet/igie/README.md#fp16)   |      |

## LLM (Large Language Model)

| Models             | vLLM                                                                | TRT-LLM                                                          | TGI                                                                                 |
|--------------------|---------------------------------------------------------------------|------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| Baichuan2-7B       | [✅](models/nlp/large_language_model/baichuan2-7b/vllm/README.md)    |                                                                  |                                                                                     |
| ChatGLM-3-6B       | [✅](models/nlp/large_language_model/chatglm3-6b/vllm/README.md)     |                                                                  |                                                                                     |
| ChatGLM-3-6B-32K   | [✅](models/nlp/large_language_model/chatglm3-6b-32k/vllm/README.md) |                                                                  |                                                                                     |
| Llama2-7B          | [✅](models/nlp/large_language_model/llama2-7b/vllm/README.md)       | [✅](models/nlp/large_language_model/llama2-7b/trtllm/README.md)  |                                                                                     |
| Llama2-13B         |                                                                     | [✅](models/nlp/large_language_model/llama2-13b/trtllm/README.md) |                                                                                     |
| Llama2-70B         |                                                                     | [✅](models/nlp/large_language_model/llama2-70b/trtllm/README.md) |                                                                                     |
| Llama3-70B         | [✅](models/nlp/large_language_model/llama3-70b/vllm/README.md)      |                                                                  |                                                                                     |
| Qwen-7B            | [✅](models/nlp/large_language_model/qwen-7b/vllm/README.md)         |                                                                  |                                                                                     |
| Qwen1.5-7B         | [✅](models/nlp/large_language_model/qwen1.5-7b/vllm/README.md)      |                                                                  | [✅](models/nlp/large_language_model/qwen1.5-7b/text-generation-inference/README.md) |
| Qwen1.5-14B        | [✅](models/nlp/large_language_model/qwen1.5-14b/vllm/README.md)     |                                                                  |                                                                                     |
| Qwen1.5-32B Chat   | [✅](models/nlp/large_language_model/qwen1.5-32b/vllm/README.md)     |                                                                  |                                                                                     |
| Qwen1.5-72B        | [✅](models/nlp/large_language_model/qwen1.5-72b/vllm/README.md)     |                                                                  |                                                                                     |
| Qwen2-7B Instruct  | [✅](models/nlp/large_language_model/qwen2-7b/vllm/README.md)        |                                                                  |                                                                                     |
| Qwen2-72B Instruct | [✅](models/nlp/large_language_model/qwen2-72b/vllm/README.md)       |                                                                  |                                                                                     |
| StableLM2-1.6B     | [✅](models/nlp/large_language_model/stablelm/vllm/README.md)        |                                                                  |                                                                                     |

## Multimodal

| Models              | vLLM                                                                                    |
|---------------------|-----------------------------------------------------------------------------------------|
| Chameleon-7B        | [✅](models/multimodal/vision_language_understanding/chameleon_7b/vllm/README.md)        |
| Fuyu-8B             | [✅](models/multimodal/vision_language_understanding/fuyu_8b/vllm/README.md)             |
| InternVL2-4B        | [✅](models/multimodal/vision_language_understanding/intern_vl/vllm/README.md)           |
| LLaVA               | [✅](models/multimodal/vision_language_understanding/llava/vllm/README.md)               |
| LLaVA-Next-Video-7B | [✅](models/multimodal/vision_language_understanding/llava_next_video_7b/vllm/README.md) |
| MiniCPM V2          | [✅](models/multimodal/vision_language_understanding/minicpm_v_2/vllm/README.md)         |

## NLP

### Language Modelling

| Models           | Precision | IGIE                                                                | IxRT                                                                |
|------------------|-----------|---------------------------------------------------------------------|---------------------------------------------------------------------|
| ALBERT           | FP16      |                                                                     | [✅](models/nlp/language_model/albert/ixrt/README.md)                |
| BERT Base NER    | INT8      | [✅](models/nlp/language_model/bert_base_ner/igie/README.md#int8)    |                                                                     |
| BERT Base SQuAD  | FP16      | [✅](models/nlp/language_model/bert_base_squad/igie/README.md#fp16)  | [✅](models/nlp/language_model/bert_base_squad/ixrt/README.md#fp16)  |
|                  | INT8      |                                                                     | [✅](models/nlp/language_model/bert_base_squad/ixrt/README.md#int8)  |
| BERT Large SQuAD | FP16      | [✅](models/nlp/language_model/bert_large_squad/igie/README.md#fp16) | [✅](models/nlp/language_model/bert_large_squad/ixrt/README.md#fp16) |
|                  | INT8      | [✅](models/nlp/language_model/bert_large_squad/igie/README.md#int8) | [✅](models/nlp/language_model/bert_large_squad/ixrt/README.md#int8) |
| DeBERTa          | FP16      |                                                                     | [✅](models/nlp/language_model/deberta/ixrt/README.md)               |
| RoBERTa          | FP16      |                                                                     | [✅](models/nlp/language_model/roberta/ixrt/README.md)               |
| RoFormer         | FP16      |                                                                     | [✅](models/nlp/language_model/roformer/ixrt/README.md)              |
| VideoBERT        | FP16      |                                                                     | [✅](models/nlp/language_model/videobert/ixrt/README.md)             |

## Speech

### Speech Recognition

| Models          | Precision | IGIE                                                                | IxRT                                                                 |
|-----------------|-----------|---------------------------------------------------------------------|----------------------------------------------------------------------|
| Conformer       | FP16      | [✅](models/speech/speech_recognition/conformer/igie/README.md#fp16) | [✅](models/speech/speech_recognition/conformer/ixrt/README.md#fp16)  |
| Transformer ASR | FP16      |                                                                     | [✅](models/speech/speech_recognition/transformer_asr/ixrt/README.md) |

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
