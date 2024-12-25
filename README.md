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
        <td><a href="models/cv/classification/clip/ixformer/README.md#fp16">ixFormer</a></td>
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
        <td rowspan=2>ConvNeXt-Base</td>
        <td>FP16</td>
        <td><a href="models/cv/classification/convnext_base/igie/README.md#fp16">Supported</a></td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td rowspan=2>ConvNeXt-Small</td>
        <td>FP16</td>
        <td><a href="models/cv/classification/convnext_small/igie/README.md#fp16">Supported</a></td>
        <td><a href="models/cv/classification/convnext_small/ixrt/README.md#fp16">Supported</a></td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td rowspan=2>CSPDarkNet53</td>
        <td>FP16</td>
        <td><a href="models/cv/classification/cspdarknet53/igie/README.md#fp16">Supported</a></td>
        <td><a href="models/cv/classification/cspdarknet53/ixrt/README.md#fp16">Supported</a></td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td>-</td>
        <td><a href="models/cv/classification/cspdarknet53/ixrt/README.md#int8">Supported</a></td>
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
        <td rowspan=2>DeiT-tiny</td>
        <td>FP16</td>
        <td><a href="models/cv/classification/deit_tiny/igie/README.md#fp16">Supported</a></td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td rowspan=2>DenseNet121</td>
        <td>FP16</td>
        <td><a href="models/cv/classification/densenet121/igie/README.md#fp16">Supported</a></td>
        <td><a href="models/cv/classification/densenet121/ixrt/README.md#fp16">Supported</a></td>
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
        <td><a href="models/cv/classification/densenet161/ixrt/README.md#fp16">Supported</a></td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td rowspan=2>DenseNet169</td>
        <td>FP16</td>
        <td><a href="models/cv/classification/densenet169/igie/README.md#fp16">Supported</a></td>
        <td><a href="models/cv/classification/densenet169/ixrt/README.md#fp16">Supported</a></td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td rowspan=2>DenseNet201</td>
        <td>FP16</td>
        <td><a href="models/cv/classification/densenet201/igie/README.md#fp16">Supported</a></td>
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
        <td rowspan=2>EfficientNet-B1</td>
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
        <td rowspan=2>EfficientNet-B2</td>
        <td>FP16</td>
        <td><a href="models/cv/classification/efficientnet_b2/igie/README.md#fp16">Supported</a></td>
        <td><a href="models/cv/classification/efficientnet_b2/ixrt/README.md#fp16">Supported</a></td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td rowspan=2>EfficientNet-B3</td>
        <td>FP16</td>
        <td><a href="models/cv/classification/efficientnet_b3/igie/README.md#fp16">Supported</a></td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td rowspan=2>EfficientNetV2</td>
        <td>FP16</td>
        <td><a href="models/cv/classification/efficientnet_v2/igie/README.md#fp16">Supported</a></td>
        <td><a href="models/cv/classification/efficientnet_v2/ixrt/README.md#fp16">Supported</a></td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td>-</td>
        <td><a href="models/cv/classification/efficientnet_v2/ixrt/README.md#int8">Supported</a></td>
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
        <td rowspan=2>EfficientNetv2_s</td>
        <td>FP16</td>
        <td><a href="models/cv/classification/efficientnet_v2_s/igie/README.md#fp16">Supported</a></td>
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
        <td><a href="models/cv/classification/hrnet_w18/ixrt/README.md#fp16">Supported</a></td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td>-</td>
        <td><a href="models/cv/classification/hrnet_w18/ixrt/README.md#int8">Supported</a></td>
    </tr>
    <tr align="center">
        <td rowspan=2>InceptionV3</td>
        <td>FP16</td>
        <td><a href="models/cv/classification/inception_v3/igie/README.md#fp16">Supported</a></td>
        <td><a href="models/cv/classification/inception_v3/ixrt/README.md#fp16">Supported</a></td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td><a href="models/cv/classification/inception_v3/igie/README.md#int8">Supported</a></td>
        <td><a href="models/cv/classification/inception_v3/ixrt/README.md#int8">Supported</a></td>
    </tr>
    <tr align="center">
        <td rowspan=2>Inception_ResNet_V2</td>
        <td>FP16</td>
        <td>-</td>
        <td><a href="models/cv/classification/inceptionresnetv2/ixrt/README.md#fp16">Supported</a></td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td>-</td>
        <td><a href="models/cv/classification/inceptionresnetv2/ixrt/README.md#int8">Supported</a></td>
    </tr>
    <tr align="center">
        <td rowspan=2>MNASNet0_5</td>
        <td>FP16</td>
        <td><a href="models/cv/classification/mnasnet0_5/igie/README.md#fp16">Supported</a></td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td>-</td>
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
        <td rowspan=2>MViTv2_base</td>
        <td>FP16</td>
        <td><a href="models/cv/classification/mvitv2_base/igie/README.md#fp16">Supported</a></td>
        <td>-</td>
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
        <td rowspan=2>RegNet_y_1_6gf</td>
        <td>FP16</td>
        <td><a href="models/cv/classification/regnet_y_1_6gf/igie/README.md#fp16">Supported</a></td>
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
        <td><a href="models/cv/classification/repvgg/igie/README.md#fp16">Supported</a></td>
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
        <td rowspan=2>ResNetV1D50</td>
        <td>FP16</td>
        <td><a href="models/cv/classification/resnetv1d50/igie/README.md#fp16">Supported</a></td>
        <td><a href="models/cv/classification/resnetv1d50/ixrt/README.md#fp16">Supported</a></td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td>-</td>
        <td><a href="models/cv/classification/resnetv1d50/ixrt/README.md#int8">Supported</a></td>
    </tr>
    <tr align="center">
        <td rowspan=2>ResNeXt50_32x4d</td>
        <td>FP16</td>
        <td><a href="models/cv/classification/resnext50_32x4d/igie/README.md#fp16">Supported</a></td>
        <td><a href="models/cv/classification/resnext50_32x4d/ixrt/README.md#fp16">Supported</a></td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td rowspan=2>ResNeXt101_64x4d</td>
        <td>FP16</td>
        <td><a href="models/cv/classification/resnext101_64x4d/igie/README.md#fp16">Supported</a></td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td rowspan=2>SEResNet50</td>
        <td>FP16</td>
        <td><a href="models/cv/classification/se_resnet50/igie/README.md#fp16">Supported</a></td>
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
        <td rowspan=2>ShuffleNetV2_x1_0</td>
        <td>FP16</td>
        <td><a href="models/cv/classification/shufflenetv2_x1_0/igie/README.md#fp16">Supported</a></td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td rowspan=2>ShuffleNetV2_x1_5</td>
        <td>FP16</td>
        <td><a href="models/cv/classification/shufflenetv2_x1_5/igie/README.md#fp16">Supported</a></td>
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
        <td rowspan=2>SqueezeNet 1.1</td>
        <td>FP16</td>
        <td>-</td>
        <td><a href="models/cv/classification/squeezenet_1.1/ixrt/README.md#fp16">Supported</a></td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td>-</td>
        <td><a href="models/cv/classification/squeezenet_1.1/ixrt/README.md#int8">Supported</a></td>
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
        <td rowspan=2>Swin Transformer Large</td>
        <td>FP16</td>
        <td>-</td>
        <td><a href="models/cv/classification/swin_transformer_large/ixrt/README.md">Supported</a></td>
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
        <td rowspan=2>Wide ResNet50</td>
        <td>FP16</td>
        <td><a href="models/cv/classification/wide_resnet50/igie/README.md#fp16">Supported</a></td>
        <td><a href="models/cv/classification/wide_resnet50/ixrt/README.md#fp16">Supported</a></td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td><a href="models/cv/classification/wide_resnet50/igie/README.md#int8">Supported</a></td>
        <td><a href="models/cv/classification/wide_resnet50/ixrt/README.md#int8">Supported</a></td>
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
        <td rowspan=2>ATSS</td>
        <td>FP16</td>
        <td><a href="models/cv/detection/atss/igie/README.md#fp16">Supported</a></td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td rowspan=2>CenterNet</td>
        <td>FP16</td>
        <td><a href="models/cv/detection/centernet/igie/README.md#fp16">Supported</a></td>
        <td><a href="models/cv/detection/centernet/ixrt/README.md#fp16">Supported</a></td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td rowspan=2>DETR</td>
        <td>FP16</td>
        <td>-</td>
        <td><a href="models/cv/detection/detr/ixrt/README.md#fp16">Supported</a></td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td rowspan=2>FCOS</td>
        <td>FP16</td>
        <td><a href="models/cv/detection/fcos/igie/README.md#fp16">Supported</a></td>
        <td><a href="models/cv/detection/fcos/ixrt/README.md#fp16">Supported</a></td>
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
        <td rowspan=2>FSAF</td>
        <td>FP16</td>
        <td><a href="models/cv/detection/fsaf/igie/README.md#fp16">Supported</a></td>
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
        <td rowspan=2>PAA</td>
        <td>FP16</td>
        <td><a href="models/cv/detection/paa/igie/README.md#fp16">Supported</a></td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td rowspan=2>RetinaFace</td>
        <td>FP16</td>
        <td><a href="models/cv/detection/retinaface/igie/README.md#fp16">Supported</a></td>
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
        <td rowspan=2>RTMDet</td>
        <td>FP16</td>
        <td><a href="models/cv/detection/rtmdet/igie/README.md#fp16">Supported</a></td>
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
        <td><a href="models/cv/detection/yolov3/ixrt/README.md#fp16">Supported</a></td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td><a href="models/cv/detection/yolov3/igie/README.md#int8">Supported</a></td>
        <td><a href="models/cv/detection/yolov3/ixrt/README.md#int8">Supported</a></td>
    </tr>
    <tr align="center">
        <td rowspan=2>YOLOv4</td>
        <td>FP16</td>
        <td><a href="models/cv/detection/yolov4/igie/README.md#fp16">Supported</a></td>
        <td><a href="models/cv/detection/yolov4/ixrt/README.md#fp16">Supported</a></td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td><a href="models/cv/detection/yolov4/igie/README.md#int816">Supported</a></td>
        <td><a href="models/cv/detection/yolov4/ixrt/README.md#int816">Supported</a></td>
    </tr>
    <tr align="center">
        <td rowspan=2>YOLOv5</td>
        <td>FP16</td>
        <td><a href="models/cv/detection/yolov5/igie/README.md#fp16">Supported</a></td>
        <td><a href="models/cv/detection/yolov5/ixrt/README.md#fp16">Supported</a></td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td><a href="models/cv/detection/yolov5/igie/README.md#int8">Supported</a></td>
        <td><a href="models/cv/detection/yolov5/ixrt/README.md#int8">Supported</a></td>
    </tr>
    <tr align="center">
        <td rowspan=2>YOLOv5s</td>
        <td>FP16</td>
        <td>-</td>
        <td><a href="models/cv/detection/yolov5s/ixrt/README.md#fp16">Supported</a></td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td>-</td>
        <td><a href="models/cv/detection/yolov5s/ixrt/README.md#int8">Supported</a></td>
    </tr>
    <tr align="center">
        <td rowspan=2>YOLOv6</td>
        <td>FP16</td>
        <td><a href="models/cv/detection/yolov6/igie/README.md#fp16">Supported</a></td>
        <td><a href="models/cv/detection/yolov6/ixrt/README.md#fp16">Supported</a></td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td>-</td>
        <td><a href="models/cv/detection/yolov6/ixrt/README.md#int8">Supported</a></td>
    </tr>
    <tr align="center">
        <td rowspan=2>YOLOv7</td>
        <td>FP16</td>
        <td><a href="models/cv/detection/yolov7/igie/README.md#fp16">Supported</a></td>
        <td><a href="models/cv/detection/yolov7/ixrt/README.md#fp16">Supported</a></td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td><a href="models/cv/detection/yolov7/igie/README.md#int8">Supported</a></td>
        <td><a href="models/cv/detection/yolov7/ixrt/README.md#int8">Supported</a></td>
    </tr>
    <tr align="center">
        <td rowspan=2>YOLOv8</td>
        <td>FP16</td>
        <td><a href="models/cv/detection/yolov8/igie/README.md#fp16">Supported</a></td>
        <td><a href="models/cv/detection/yolov8/ixrt/README.md#fp16">Supported</a></td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td><a href="models/cv/detection/yolov8/igie/README.md#int8">Supported</a></td>
        <td><a href="models/cv/detection/yolov8/ixrt/README.md#int8">Supported</a></td>
    </tr>
    <tr align="center">
        <td rowspan=2>YOLOv9</td>
        <td>FP16</td>
        <td><a href="models/cv/detection/yolov9/igie/README.md#fp16">Supported</a></td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td rowspan=2>YOLOv10</td>
        <td>FP16</td>
        <td><a href="models/cv/detection/yolov10/igie/README.md#fp16">Supported</a></td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td>-</td>
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

### Face Recognition

<table>
    <tr align="center">
        <th>Models</th>
        <th>Precision</th>
        <th>IGIE</th>
        <th>IxRT</th>
    </tr>
    <tr align="center">
        <td rowspan=2>FaceNet
</td>
        <td>FP16</td>
        <td>-</td>
        <td><a href="models/cv/face/facenet/ixrt/README.md#fp16">Supported</a></td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td>-</td>
        <td><a href="models/cv/face/facenet/ixrt/README.md#int8">Supported</a></td>
    </tr>
</table>

### OCR

<table>
    <tr align="center">
        <th>Models</th>
        <th>Precision</th>
        <th>IGIE</th>
        <th>IxRT</th>
    </tr>
    <tr align="center">
        <td rowspan=2>Kie_layoutXLM</td>
        <td>FP16</td>
        <td><a href="models/cv/ocr/kie_layoutxlm/igie/README.md#fp16">Supported</a></td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td rowspan=2>Rec_SVTR</td>
        <td>FP16</td>
        <td><a href="models/cv/ocr/rec_svtr/igie/README.md#fp16">Supported</a></td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td>-</td>
        <td>-</td>
    </tr>
</table>

### Pose Estimation

<table>
    <tr align="center">
        <th>Models</th>
        <th>Precision</th>
        <th>IGIE</th>
        <th>IxRT</th>
    </tr>
    <tr align="center">
        <td rowspan=2>Lightweight OpenPose</td>
        <td>FP16</td>
        <td>-</td>
        <td><a href="models/cv/pose_estimation/lightweightopenpose/ixrt/README.md#fp16">Supported</a></td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td rowspan=2>RTMPose</td>
        <td>FP16</td>
        <td><a href="models/cv/pose_estimation/rtmpose/igie/README.md#fp16">Supported</a></td>
        <td><a href="models/cv/pose_estimation/rtmpose/ixrt/README.md#fp16">Supported</a></td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td>-</td>
        <td>-</td>
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
    <tr align="center">
        <td rowspan=2>SOLOv1</td>
        <td>FP16</td>
        <td>-</td>
        <td><a href="models/cv/segmentation/solov1/ixrt/README.md#fp16">Supported</a></td>
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

### Language Modelling

<table>
    <tr align="center">
        <th>Models</th>
        <th>Precision</th>
        <th>IGIE</th>
        <th>IxRT</th>
    </tr>
    <tr align="center">
        <td rowspan=2>ALBERT</td>
        <td>FP16</td>
        <td>-</td>
        <td><a href="models/nlp/language_model/albert/ixrt/README.md">Supported</a></td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td>-</td>
        <td>-</td>
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
        <td><a href="models/nlp/language_model/bert_base_squad/ixrt/README.md#int8">Supported</a></td>
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
    <tr align="center">
        <td rowspan=2>DeBERTa</td>
        <td>FP16</td>
        <td>-</td>
        <td><a href="models/nlp/language_model/deberta/ixrt/README.md">Supported</a></td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td rowspan=2>RoBERTa</td>
        <td>FP16</td>
        <td>-</td>
        <td><a href="models/nlp/language_model/roberta/ixrt/README.md">Supported</a></td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td rowspan=2>RoFormer</td>
        <td>FP16</td>
        <td>-</td>
        <td><a href="models/nlp/language_model/roformer/ixrt/README.md">Supported</a></td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td rowspan=2>VideoBERT</td>
        <td>FP16</td>
        <td>-</td>
        <td><a href="models/nlp/language_model/videobert/ixrt/README.md">Supported</a></td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td>-</td>
        <td>-</td>
    </tr>
</table>

### Large Language Model

<table>
    <tr align="center">
        <th>Models</th>
        <th>vLLM</th>
        <th>TensorRT-LLM</th>
        <th>TGI</th>
    </tr>
    <tr align="center">
        <td>Baichuan2-7B</td>
        <td><a href="models/nlp/large_language_model/baichuan2-7b/vllm/README.md">Supported</a></td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td>ChatGLM-3-6B</td>
        <td><a href="models/nlp/large_language_model/chatglm3-6b/vllm/README.md">Supported</a></td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td>ChatGLM-3-6B-32K</td>
        <td><a href="models/nlp/large_language_model/chatglm3-6b-32k/vllm/README.md">Supported</a></td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td>Llama2-7B</td>
        <td><a href="models/nlp/large_language_model/llama2-7b/vllm/README.md">Supported</a></td>
        <td><a href="models/nlp/large_language_model/llama2-7b/trtllm/README.md">Supported</a></td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td>Llama3-70B</td>
        <td><a href="models/nlp/large_language_model/llama3-70b/vllm/README.md">Supported</a></td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td>MiniCPM-V-2</td>
        <td><a href="models/vision-language-understanding/MiniCPM-V-2/vllm/README.md">Supported</a></td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td>Qwen-7B</td>
        <td><a href="models/nlp/large_language_model/qwen-7b/vllm/README.md">Supported</a></td>
        <td>-</td>
        <td><a href="models/nlp/large_language_model/qwen-7b/text-generation-inference/README.md">Supported</a></td>
    </tr>
    <tr align="center">
        <td>Qwen1.5-7B</td>
        <td><a href="models/nlp/large_language_model/qwen1.5-7b/vllm/README.md">Supported</a></td>
        <td>-</td>
        <td><a href="models/nlp/large_language_model/qwen1.5-7b/text-generation-inference/README.md">Supported</a></td>
    </tr>
    <tr align="center">
        <td>Qwen1.5-14B</td>
        <td><a href="models/nlp/large_language_model/qwen1.5-14b/vllm/README.md">Supported</a></td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td>Qwen1.5-32B Chat</td>
        <td><a href="models/nlp/large_language_model/qwen1.5-32b/vllm/README.md">Supported</a></td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td>Qwen1.5-72B</td>
        <td><a href="models/nlp/large_language_model/qwen1.5-72b/vllm/README.md">Supported</a></td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td>Qwen2-7B Instruct</td>
        <td><a href="models/nlp/large_language_model/qwen2-7b/vllm/README.md">Supported</a></td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td>Qwen2-72B Instruct</td>
        <td><a href="models/nlp/large_language_model/qwen2-72b/vllm/README.md">Supported</a></td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr align="center">
        <td>StableLM2-1.6B</td>
        <td><a href="models/nlp/large_language_model/stablelm/vllm/README.md">Supported</a></td>
        <td>-</td>
        <td>-</td>
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
        <td><a href="models/speech/speech_recognition/conformer/ixrt/README.md#fp16">Supported</a></td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td>-</td>
        <td>-</td>
    </tr>
        <tr align="center">
        <td rowspan=2>Transformer ASR</td>
        <td>FP16</td>
        <td>-</td>
        <td><a href="models/speech/speech_recognition/transformer_asr/ixrt/README.md">Supported</a></td>
    </tr>
    <tr align="center">
        <td>INT8</td>
        <td>-</td>
        <td>-</td>
    </tr>
</table>

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
