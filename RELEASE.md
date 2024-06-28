# DeepSparkInference Release Notes

## 24.06 Release Notes

### 模型与算法

* 新增了31个推理小模型示例，其中支持IGIE推理引擎的16个，支持IxRT推理引擎的15个。

<table>
    <tr align="left">
        <th colspan=6>IGIE</th>
    </tr>
    <tr>
        <td>CenterNet</td>
        <td>DenseNet161</td>
        <td>EfficientNet_b1</td>
        <td>EfficientNet_rw_t</td>
    </tr>
    <tr>
        <td>FoveaBox</td>
        <td>HRNet</td>
        <td>MobileNetv3_Large</td>
        <td>MobileNetv3_Small</td>
    </tr>
    <tr>
        <td>Regnet_x_1_6gf</td>
        <td>RepNet</td>
        <td>Res2Net50</td>
        <td>ResNeSt50</td>
    </tr>
    <tr>
        <td>ResNet101</td>
        <td>ResNet152</td>
        <td>ShuffleNetv2_x0_5</td>
        <td>Wide_ResNet50</td>
    </tr>
    <tr align="left">
        <th colspan=6>IxRT</th>
    </tr>
    <tr>
        <td>BERT Base SQuAD</td>
        <td>DenseNet</td>
        <td>DETR</td>
        <td>EfficientNet_V2_t</td>
    </tr>
    <tr>
        <td>FCOS</td>
        <td>HRNet</td>
        <td>Inception_ResNet_V2</td>
        <td>InceptionV3</td>
    </tr>
    <tr>
        <td>ResNet_V1_D50</td>
        <td>SoloV1</td>
        <td>SqueezeNet_v1.1</td>
        <td>YOLOv3</td>
    </tr>
    <tr>
        <td>YOLOv5m</td>
        <td>YOLOv5s</td>
        <td>YOLOv7</td>
        <td></td>
    </tr>
</table>

* 新增了4个大语言模型的推理示例。
  * Baichuan2-7B (vLLM)
  * ChatGLM-3-6B (vLLM)
  * Llama2-7B (TensorRT-LLM)
  * Qwen-7B (Text Generation Inference)

### 问题修复

* 修复了YOLOX IxRT 插件编译会报错的问题。
* 完善了libGL依赖在Ubuntu上安装的帮助说明。

### 版本关联

DeepSparkInference 24.06对应天数软件栈4.0.0版本。

### 感谢以下社区贡献者

YoungPeng，majorli，xinchi.tian，tianxi-yi，may，xiaomei.wang，xiaomei.wang，cheneychen2023，qiang.zhang。

---

## 24.03 Release Notes

### 模型与算法

* 新增了48个推理模型示例，其中支持IGIE推理引擎的28个，支持IxRT推理引擎的20个。

<table>
    <tr align="left">
        <th colspan=6>IGIE</th>
    </tr>
    <tr>
        <td>AlexNet</td>
        <td>BERT Base NER</td>
        <td>BERT Base SQuAD</td>
        <td>BERT Large SQuAD</td>
    </tr>
    <tr>
        <td>CLIP</td>
        <td>Conformer</td>
        <td>Conformer-B</td>
        <td>DeepSort</td>
    </tr>
    <tr>
        <td>DenseNet121</td>
        <td>EfficientNet-B0</td>
        <td>FastReID</td>
        <td>GoogLeNet</td>
    </tr>
    <tr>
        <td>HRNet-W18</td>
        <td>InceptionV3</td>
        <td>MobileNetV2</td>
        <td>ResNet18</td>
    </tr>
    <tr>
        <td>ResNet50</td>
        <td>ResNeXt50_32x4d</td>
        <td>RetinaNet</td>
        <td>Swin Transformer</td>
    </tr>
    <tr>
        <td>VGG16</td>
        <td>YOLOv3</td>
        <td>YOLOv4</td>
        <td>YOLOv5</td>
    </tr>
    <tr>
        <td>YOLOv6</td>
        <td>YOLOv7</td>
        <td>YOLOv8</td>
        <td>YOLOX</td>
    </tr>
    <tr align="left">
        <th colspan=6>IxRT</th>
    </tr>
    <tr>
        <td>AlexNet</td>
        <td>BERT Base SQuAD</td>
        <td>BERT Large SQuAD</td>
        <td>CSPResNet50</td>
    </tr>
    <tr>
        <td>EfficientNet-B0</td>
        <td>EfficientNet-B1</td>
        <td>GoogLeNet</td>
        <td>Mask R-CNN</td>
    </tr>
    <tr>
        <td>MobileNetV2</td>
        <td>MobileNetV3</td>
        <td>RepVGG</td>
        <td>Res2Net50</td>
    </tr>
    <tr>
        <td>ResNet101</td>
        <td>ResNet18</td>
        <td>ResNet34</td>
        <td>ResNet50</td>
    </tr>
    <tr>
        <td>ShufflenetV1</td>
        <td>SqueezeNet 1.0</td>
        <td>VGG16</td>
        <td>YOLOX</td>
</table>

### 版本关联

DeepSparkInference 24.03对应天数软件栈4.0.0版本。
