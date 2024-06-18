# DeepSparkInference

DeepSparkInference推理模型库作为DeepSpark开源社区的核心项目，于2024年3月正式开源，一期甄选了48个推理模型示例，涵盖计算机视觉，自然语言处理，语音识别等领域，后续将逐步拓展更多AI领域。

DeepSparkInference中的模型提供了在国产推理引擎IGIE或IxRT下运行的推理示例和指导文档，部分模型提供了基于国产通用GPU[智铠100](https://www.iluvatar.com/productDetails?fullCode=cpjs-yj-tlxltt-zk100)的评测结果。

IGIE（Iluvatar GPU Inference Engine）是基于TVM框架研发的高性能、高通用、全流程的AI推理引擎。支持多框架模型导入、量化、图优化、多算子库支持、多后端支持、算子自动调优等特性，为推理场景提供易部署、高吞吐量、低延迟的完整方案。

IxRT（Iluvatar CoreX RunTime）是天数智芯自研的高性能推理引擎，专注于最大限度发挥天数智芯通用GPU 的性能，实现各领域模型的高性能推理。IxRT支持动态形状推理、插件和INT8/FP16推理等特性。

DeepSparkInference将按季度进行版本更新，后续会逐步丰富模型类别并拓展大模型推理。

## Computer Vision

### Classification

<table>
    <tr align="center">
        <th>Models</th>
        <th>Precision</th>
        <th>IGIE</th>
        <th>IxRT</th>
    </tr>
    <tr align="center">
        <td rowspan=2>AlexNet</td>
        <td>FP16</td>
        <td><a href="models/cv/classification/alexnet/igie/README.md#fp16">Supported</a></td>
        <td><a href="models/cv/classification/alexnet/ixrt/README.md#fp16">Supported</a></td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td><a href="models/cv/classification/alexnet/igie/README.md#int8">Supported</a></td>
        <td><a href="models/cv/classification/alexnet/ixrt/README.md#int8">Supported</a></td>
    </tr>
    <tr align="center">
        <td rowspan=2>CLIP</td>
        <td>FP16</td>
        <td><a href="models/cv/classification/clip/igie/README.md#fp16">Supported</a></td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td rowspan=2>Conformer-B</td>
        <td>FP16</td>
        <td><a href="models/cv/classification/conformer_base/igie/README.md#fp16">Supported</a></td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td>-</td>
        <td>-</td>
    </tr>
        <tr align="center">
        <td rowspan=2>CSPResNet50</td>
        <td>FP16</td>
        <td>-</td>
        <td><a href="models/cv/classification/cspresnet50/ixrt/README.md#fp16">Supported</a></td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td>-</td>
        <td><a href="models/cv/classification/cspresnet50/ixrt/README.md#int8">Supported</a></td>
    </tr>
    <tr align="center">
        <td rowspan=2>DenseNet121</td>
        <td>FP16</td>
        <td><a href="models/cv/classification/densenet121/igie/README.md#fp16">Supported</a></td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td rowspan=2>DenseNet161</td>
        <td>FP16</td>
        <td><a href="models/cv/classification/densenet161/igie/README.md#fp16">Supported</a></td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td rowspan=2>EfficientNet-B0</td>
        <td>FP16</td>
        <td><a href="models/cv/classification/efficientnet_b0/igie/README.md#fp16">Supported</a></td>
        <td><a href="models/cv/classification/efficientnet_b0/ixrt/README.md#fp16">Supported</a></td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td>-</td>
        <td><a href="models/cv/classification/efficientnet_b0/ixrt/README.md#int8">Supported</a></td>
    </tr>
    <tr align="center">
        <td rowspan=2>EfficientNet_B1</td>
        <td>FP16</td>
        <td><a href="models/cv/classification/efficientnet_b1/igie/README.md#fp16">Supported</a></td>
        <td><a href="models/cv/classification/efficientnet_b1/ixrt/README.md#fp16">Supported</a></td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td>-</td>
        <td><a href="models/cv/classification/efficientnet_b1/ixrt/README.md#int8">Supported</a></td>
    </tr>
    <tr align="center">
        <td rowspan=2>EfficientNetv2_rw_t</td>
        <td>FP16</td>
        <td><a href="models/cv/classification/efficientnetv2_rw_t/igie/README.md#fp16">Supported</a></td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td rowspan=2>GoogLeNet</td>
        <td>FP16</td>
        <td><a href="models/cv/classification/googlenet/igie/README.md#fp16">Supported</a></td>
        <td><a href="models/cv/classification/googlenet/ixrt/README.md#fp16">Supported</a></td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td><a href="models/cv/classification/googlenet/igie/README.md#int8">Supported</a></td>
        <td><a href="models/cv/classification/googlenet/ixrt/README.md#int8">Supported</a></td>
    </tr>
    <tr align="center">
        <td rowspan=2>HRNet-W18</td>
        <td>FP16</td>
        <td><a href="models/cv/classification/hrnet_w18/igie/README.md#fp16">Supported</a></td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td rowspan=2>InceptionV3</td>
        <td>FP16</td>
        <td><a href="models/cv/classification/inception_v3/igie/README.md#fp16">Supported</a></td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td><a href="models/cv/classification/inception_v3/igie/README.md#int8">Supported</a></td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td rowspan=2>MobileNetV2</td>
        <td>FP16</td>
        <td><a href="models/cv/classification/mobilenet_v2/igie/README.md#fp16">Supported</a></td>
        <td><a href="models/cv/classification/mobilenet_v2/ixrt/README.md#fp16">Supported</a></td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td><a href="models/cv/classification/mobilenet_v2/igie/README.md#int8">Supported</a></td>
        <td><a href="models/cv/classification/mobilenet_v2/ixrt/README.md#int8">Supported</a></td>
    </tr>
    <tr align="center">
        <td rowspan=2>MobileNetV3_Large</td>
        <td>FP16</td>
        <td><a href="models/cv/classification/mobilenet_v3_large/igie/README.md#fp16">Supported</a></td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td rowspan=2>MobileNetV3_Small</td>
        <td>FP16</td>
        <td><a href="models/cv/classification/mobilenet_v3/igie/README.md#fp16">Supported</a></td>
        <td><a href="models/cv/classification/mobilenet_v3/ixrt/README.md#fp16">Supported</a></td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td rowspan=2>RegNet_x_1_6gf</td>
        <td>FP16</td>
        <td><a href="models/cv/classification/regnet_x_1_6gf/igie/README.md#fp16">Supported</a></td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td rowspan=2>RepVGG</td>
        <td>FP16</td>
        <td>-</td>
        <td><a href="models/cv/classification/repvgg/ixrt/README.md#fp16">Supported</a></td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td rowspan=2>Res2Net50</td>
        <td>FP16</td>
        <td><a href="models/cv/classification/res2net50/igie/README.md#fp16">Supported</a></td>
        <td><a href="models/cv/classification/res2net50/ixrt/README.md#fp16">Supported</a></td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td>-</td>
        <td><a href="models/cv/classification/res2net50/ixrt/README.md#int8">Supported</a></td>
    </tr>
    <tr align="center">
        <td rowspan=2>ResNeSt50</td>
        <td>FP16</td>
        <td><a href="models/cv/classification/resnest50/igie/README.md#fp16">Supported</a></td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td rowspan=2>ResNet101</td>
        <td>FP16</td>
        <td><a href="models/cv/classification/resnet101/igie/README.md#fp16">Supported</a></td>
        <td><a href="models/cv/classification/resnet101/ixrt/README.md#fp16">Supported</a></td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td><a href="models/cv/classification/resnet101/igie/README.md#int8">Supported</a></td>
        <td><a href="models/cv/classification/resnet101/ixrt/README.md#int8">Supported</a></td>
    </tr>
    <tr align="center">
        <td rowspan=2>ResNet152</td>
        <td>FP16</td>
        <td><a href="models/cv/classification/resnet152/igie/README.md#fp16">Supported</a></td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td><a href="models/cv/classification/resnet152/igie/README.md#int8">Supported</a></td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td rowspan=2>ResNet18</td>
        <td>FP16</td>
        <td><a href="models/cv/classification/resnet18/igie/README.md#fp16">Supported</a></td>
        <td><a href="models/cv/classification/resnet18/ixrt/README.md#fp16">Supported</a></td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td><a href="models/cv/classification/resnet18/igie/README.md#int8">Supported</a></td>
        <td><a href="models/cv/classification/resnet18/ixrt/README.md#int8">Supported</a></td>
    </tr>
    <tr align="center">
        <td rowspan=2>ResNet34</td>
        <td>FP16</td>
        <td>-</td>
        <td><a href="models/cv/classification/resnet34/ixrt/README.md#fp16">Supported</a></td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td>-</td>
        <td><a href="models/cv/classification/resnet34/ixrt/README.md#int8">Supported</a></td>
    </tr>
    <tr align="center">
        <td rowspan=2>ResNet50</td>
        <td>FP16</td>
        <td><a href="models/cv/classification/resnet50/igie/README.md#fp16">Supported</a></td>
        <td><a href="models/cv/classification/resnet50/ixrt/README.md#fp16">Supported</a></td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td><a href="models/cv/classification/resnet50/igie/README.md#int8">Supported</a></td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td rowspan=2>ResNeXt50_32x4d</td>
        <td>FP16</td>
        <td><a href="models/cv/classification/resnext50_32x4d/igie/README.md#fp16">Supported</a></td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td rowspan=2>ShuffleNetV1</td>
        <td>FP16</td>
        <td>-</td>
        <td><a href="models/cv/classification/shufflenet_v1/ixrt/README.md#fp16">Supported</a></td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td rowspan=2>ShuffleNetV2_x0_5</td>
        <td>FP16</td>
        <td><a href="models/cv/classification/shufflenetv2_x0_5/igie/README.md#fp16">Supported</a></td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td rowspan=2>SqueezeNet 1.0</td>
        <td>FP16</td>
        <td>-</td>
        <td><a href="models/cv/classification/squeezenet_1.0/ixrt/README.md#fp16">Supported</a></td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td>-</td>
        <td><a href="models/cv/classification/squeezenet_1.0/ixrt/README.md#int8">Supported</a></td>
    </tr>
    <tr align="center">
        <td rowspan=2>Swin Transformer</td>
        <td>FP16</td>
        <td><a href="models/cv/classification/swin_transformer/igie/README.md#fp16">Supported</a></td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td rowspan=2>VGG16</td>
        <td>FP16</td>
        <td><a href="models/cv/classification/vgg16/igie/README.md#fp16">Supported</a></td>
        <td><a href="models/cv/classification/vgg16/ixrt/README.md#fp16">Supported</a></td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td><a href="models/cv/classification/vgg16/igie/README.md#int8">Supported</a></td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td rowspan=2>Wide_ResNet50</td>
        <td>FP16</td>
        <td><a href="models/cv/classification/wide_resnet50/igie/README.md#fp16">Supported</a></td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td><a href="models/cv/classification/wide_resnet50/igie/README.md#int8">Supported</a></td>
        <td>-</td>
    </tr>
</table>

### Detection

<table>
    <tr align="center">
        <th>Models</th>
        <th>Precision</th>
        <th>IGIE</th>
        <th>IxRT</th>
    </tr>
    <tr align="center">
        <td rowspan=2>CenterNet</td>
        <td>FP16</td>
        <td><a href="models/cv/detection/centernet/igie/README.md#fp16">Supported</a></td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td rowspan=2>FoveaBox</td>
        <td>FP16</td>
        <td><a href="models/cv/detection/foveabox/igie/README.md#fp16">Supported</a></td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td rowspan=2>HRNet</td>
        <td>FP16</td>
        <td><a href="models/cv/detection/hrnet/igie/README.md#fp16">Supported</a></td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td rowspan=2>RetinaNet</td>
        <td>FP16</td>
        <td><a href="models/cv/detection/retinanet/igie/README.md#fp16">Supported</a></td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td rowspan=2>YOLOv3</td>
        <td>FP16</td>
        <td><a href="models/cv/detection/yolov3/igie/README.md#fp16">Supported</a></td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td><a href="models/cv/detection/yolov3/igie/README.md#int8">Supported</a></td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td rowspan=2>YOLOv4</td>
        <td>FP16</td>
        <td><a href="models/cv/detection/yolov4/igie/README.md#fp16">Supported</a></td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td><a href="models/cv/detection/yolov4/igie/README.md#int816">Supported</a></td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td rowspan=2>YOLOv5</td>
        <td>FP16</td>
        <td><a href="models/cv/detection/yolov5/igie/README.md#fp16">Supported</a></td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td><a href="models/cv/detection/yolov5/igie/README.md#int8">Supported</a></td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td rowspan=2>YOLOv6</td>
        <td>FP16</td>
        <td><a href="models/cv/detection/yolov6/igie/README.md#fp16">Supported</a></td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td rowspan=2>YOLOv7</td>
        <td>FP16</td>
        <td><a href="models/cv/detection/yolov7/igie/README.md#fp16">Supported</a></td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td><a href="models/cv/detection/yolov7/igie/README.md#int8">Supported</a></td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td rowspan=2>YOLOv8</td>
        <td>FP16</td>
        <td><a href="models/cv/detection/yolov8/igie/README.md#fp16">Supported</a></td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td><a href="models/cv/detection/yolov8/igie/README.md#int8">Supported</a></td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td rowspan=2>YOLOX</td>
        <td>FP16</td>
        <td><a href="models/cv/detection/yolox/igie/README.md#fp16">Supported</a></td>
        <td><a href="models/cv/detection/yolox/ixrt/README.md#fp16">Supported</a></td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td><a href="models/cv/detection/yolox/igie/README.md#int8">Supported</a></td>
        <td><a href="models/cv/detection/yolox/ixrt/README.md#int8">Supported</a></td>
    </tr>
</table>

### Segmentation

<table>
    <tr align="center">
        <th>Models</th>
        <th>Precision</th>
        <th>IGIE</th>
        <th>IxRT</th>
    </tr>
    <tr align="center">
        <td rowspan=2>Mask R-CNN</td>
        <td>FP16</td>
        <td>-</td>
        <td><a href="models/cv/segmentation/mask_rcnn/ixrt/README.md#fp16">Supported</a></td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td>-</td>
        <td>-</td>
    </tr>
</table>

### Trace

<table>
    <tr align="center">
        <th>Models</th>
        <th>Precision</th>
        <th>IGIE</th>
        <th>IxRT</th>
    </tr>
    <tr align="center">
        <td rowspan=2>FastReID</td>
        <td>FP16</td>
        <td><a href="models/cv/trace/fastreid/igie/README.md#fp16">Supported</a></td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td rowspan=2>DeepSort</td>
        <td>FP16</td>
        <td><a href="models/cv/trace/deepsort/igie/README.md#fp16">Supported</a></td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td><a href="models/cv/trace/deepsort/igie/README.md#int8">Supported</a></td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td rowspan=2>RepNet-Vehicle-ReID</td>
        <td>FP16</td>
        <td><a href="models/cv/trace/repnet/igie/README.md#fp16">Supported</a></td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td>-</td>
        <td>-</td>
    </tr>
</table>

## NLP

### Language Model

<table>
    <tr align="center">
        <th>Models</th>
        <th>Precision</th>
        <th>IGIE</th>
        <th>IxRT</th>
    </tr>
    <tr align="center">
        <td rowspan=2>BERT Base NER</td>
        <td>FP16</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td><a href="models/nlp/language_model/bert_base_ner/igie/README.md#int8">Supported</a></td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td rowspan=2>BERT Base SQuAD</td>
        <td>FP16</td>
        <td><a href="models/nlp/language_model/bert_base_squad/igie/README.md#fp16">Supported</a></td>
        <td><a href="models/nlp/language_model/bert_base_squad/ixrt/README.md#fp16">Supported</a></td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td rowspan=2>BERT Large SQuAD</td>
        <td>FP16</td>
        <td><a href="models/nlp/language_model/bert_large_squad/igie/README.md#fp16">Supported</a></td>
        <td><a href="models/nlp/language_model/bert_large_squad/ixrt/README.md#fp16">Supported</a></td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td><a href="models/nlp/language_model/bert_large_squad/igie/README.md#int8">Supported</a></td>
        <td><a href="models/nlp/language_model/bert_large_squad/ixrt/README.md#int8">Supported</a></td>
    </tr>
</table>

## Speech

### Speech Recognition

<table>
    <tr align="center">
        <th>Models</th>
        <th>Precision</th>
        <th>IGIE</th>
        <th>IxRT</th>
    </tr>
    <tr align="center">
        <td rowspan=2>Conformer</td>
        <td>FP16</td>
        <td><a href="models/speech/speech_recognition/conformer/igie/README.md#fp16">Supported</a></td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td>-</td>
        <td>-</td>
    </tr>
</table>

------

## 社区

### 治理

请参见 DeepSpark Code of Conduct on [Gitee](https://gitee.com/deep-spark/deepspark/blob/master/CODE_OF_CONDUCT.md) or on [GitHub](https://github.com/Deep-Spark/deepspark/blob/main/CODE_OF_CONDUCT.md)。

### 交流

请联系 contact@deepspark.org.cn。

### 贡献

请参见 [DeepSparkInference Contributing Guidelines](CONTRIBUTING.md)。

## 许可证

本项目许可证遵循[Apache-2.0](LICENSE)。
