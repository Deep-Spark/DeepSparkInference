# DeepSparkInference Release Notes

## 25.06 Release Notes

### 模型与算法

* 新增了24个推理小模型示例，其中支持IGIE推理引擎的15个，支持IxRT推理引擎的9个。
* 新增了6个基于vLLM的大语言模型推理示例，其中3个为多模态模型。

<table>
  <tr colspan=4>
  <th colspan=3>IGIE</th>
  </tr>
  <tr>
    <td>ConvNext_Tiny</td>
    <td>CSPResNeXt50</td>
    <td>EfficientNet_B5</td>
  </tr>
  <tr>
    <td>GFL</td>
    <td>MNASNet1_0</td>
    <td>Regnet_x_3_2gf</td>
  </tr>
  <tr>
    <td>Regnet_y_16gf</td>
    <td>SqueezeNet1_1</td>
    <td>Twins_PCPVT</td>
  </tr>
  <tr>
    <td>UNet</td>
    <td>VAN_B0</td>
    <td>VGG19</td>
  </tr>
  <tr>
    <td>ViT</td>
    <td>VGG19_BN</td>
    <td>YOLOv12</td>
  </tr>
    <th colspan=4>IxRT</th>
  </tr>
  <tr>
    <td>ResNeXt101_32x8d</td>
    <td>ResNeXt101_64x4d</td>
    <td>ShuffleNetV2_x0_5</td>
  </tr>
  <tr>
    <td>ShuffleNetV2_x1_0</td>
    <td>ShuffleNetV2_x1_5</td>
    <td>ShuffleNetV2_x2_0</td>
  </tr>
  <tr>
    <td>YOLOv9</td>
    <td>YOLOv10</td>
    <td>YOLOv11</td>
  </tr>
    <th colspan=4>大模型</th>
  </tr>
  <tr>
    <td>Aria (vLLM)</td>
    <td>H2OVLChatModel (vLLM)</td>
    <td>Idefics3-8B-Llama3 (vLLM)</td>
  </tr>
  <tr>
    <td>Llama-3.2 (vLLM)</td>
    <td>MiniCPM-V2 (vLLM)</td>
    <td>Pixtral-12B-2409 (vLLM)</td>
  </tr>
</table>

### 修复更新

* 修复了模型代码中的绝对路径的问题。
* 修复了EfficientNetV2模型依赖timm版本的问题。
* 修复了GoogleNet和InceptionV3模型（IGIE）的编译报错问题。
* 更新了stable-diffusion-v1-5模型在huggingface上的链接。
* 更新了MiniCPM-V2推理模型代码示例。
* 优化了IxRT和IGIE中视觉分类模型的公共代码避免重复。
* 增加了DeepSparkInference全部模型的json配置文件。
* 增加了模型库首页的英文版README文档。

### 版本关联

DeepSparkInference 25.06对应天数软件栈4.2.0版本。

### 感谢以下社区贡献者

YoungPeng，majorli6，honglyua，qiang.zhang。

## 25.03 Release Notes

### 模型与算法

* 新增了25个推理小模型示例，其中支持IGIE推理引擎的15个，支持IxRT推理引擎的10个。
* 新增了11个大模型推理示例，其中6个为DeepSeek-R1蒸馏模型。

<table>
  <tr colspan=4>
  <th colspan=3>IGIE</th>
  </tr>
  <tr>
    <td>CSPResNet50</td>
    <td>ConvNeXt_S</td>
    <td>EfficientNet_b4</td>
  </tr>
  <tr>
    <td>HRNetPose</td>
    <td>MNASNet0_75</td>
    <td>Mixer_B</td>
  </tr>
  <tr>
    <td>Regnet_x_16gf</td>
    <td>ResNeXt101_32x8d</td>
    <td>SABL</td>
  </tr>
  <tr>
    <td>ShuffleNetv2_x2_0</td>
    <td>SqueezeNet1_0</td>
    <td>SVT_base</td>
  </tr>
  <tr>
    <td>VGG11</td>
    <td>Wide_ResNet101</td>
    <td>YOLOv11</td>
  </tr>
    <th colspan=4>IxRT</th>
  </tr>
  <tr>
    <td>ConvNeXt-Base</td>
    <td>DeiT-tiny</td>
    <td>DenseNet201</td>
  </tr>
  <tr>
    <td>EfficientNet-B3</td>
    <td>EfficientNetv2_rw_t</td>
    <td>EfficientNetv2_s</td>
  </tr>
  <tr>
    <td>FoveaBox</td>
    <td>FSAF</td>
    <td>HRNet</td>
  </tr>
  <tr>
    <td>RetinaFace</td>
    <td></td>
    <td></td>
  </tr>
    <th colspan=4>大模型</th>
  </tr>
  <tr>
    <td>DeepSeek-R1-Distill-Llama-8B</td>
    <td>DeepSeek-R1-Distill-Llama-70B</td>
    <td>DeepSeek-R1-Distill-Qwen-1.5B</td>
  </tr>
  <tr>
    <td>DeepSeek-R1-Distill-Qwen-7B</td>
    <td>DeepSeek-R1-Distill-Qwen-14B</td>
    <td>DeepSeek-R1-Distill-Qwen-32B</td>
  </tr>
  <tr>
    <td>Chameleon-7B</td>
    <td>Fuyu-8B</td>
    <td>InternVL2-4B</td>
  </tr>
  <tr>
    <td>LLaVA</td>
    <td>LLaVA-Next-Video-7B</td>
    <td></td>
  </tr>
</table>

### 问题修复

* 新增了IxRT的NLP推理模型的自动化测试运行脚本。
* 优化了IGIE推理模型自动化测试的运行脚本。
* 修复了onnxruntime 1.17.1导致的quantize fail问题。
* 修复了BERT模型在INT8精度下的问题。
* 修复了YOLOv6 IGIE模型运行推理脚本报错的问题。
* 修复了mmpretraino 0.24.0与mmcv 2.1.0版本不兼容问题。
* 修复了ChatGLM3-6B-32k模型中的中文乱码问题。
* 修复了vLLM模型中SamplingParams的初始化问题。
* 更新了所有模型README文档格式，补充了模型所支持的IXUCA SDK版本。

### 版本关联

DeepSparkInference 25.03对应天数软件栈4.2.0版本。

### 感谢以下社区贡献者

YoungPeng，majorli6，xinchi.tian，xiaomei.wang，honglyua，qiang.zhang。

## 24.12 Release Notes

### 模型与算法

* 新增了24个推理小模型示例，其中支持IGIE推理引擎的15个，支持IxRT推理引擎的9个。
* 新增了9个大语言模型的推理示例，其中支持vLLM的8个，支持ixFormer的1个。

<table>
  <tr align="left">
  <th colspan=3>IGIE</th>
  </tr>
  <tr>
    <td>ConvNeXt-Base</td>
    <td>DenseNet201</td>
    <td>EfficientNet-B3</td>
  </tr>
  <tr>
    <td>EfficientNetV2-S</td>
    <td>MNASNet0_5</td>
    <td>MViTv2_base</td>
  </tr>
  <tr>
    <td>Regnet_y_1_6gf</td>
    <td>ResNetV1_D50</td>
    <td>ResNeXt101_64x4d</td>
  </tr>
  <tr>
    <td>ShuffleNetV2_x1_5</td>
    <td>Kie_layoutXLM</td>
    <td>SVTR</td>
  </tr>
  <tr>
    <td>YOLOv9</td>
    <td>YOLOv10</td>
    <td>PAA</td>
  </tr>
    <th colspan=4>IxRT</th>
  </tr>
  <tr>
    <td>CenterNet</td>
    <td>OpenPose</td>
    <td>RTMPose</td>
  </tr>
  <tr>
    <td>CSPDarkNet53</td>
    <td>DensNet161</td>
    <td>DensNet169</td>
  </tr>
  <tr>
    <td>EfficientNetB2</td>
    <td>ResNeXt50_32x4d</td>
    <td>ConvNeXt-Small</td>
  </tr>
  </tr>
    <th colspan=4>大模型推理</th>
  </tr>
  <tr>
    <td>CLIP (IxFormer)</td>
    <td>ChatGLM3-6B-32K (vLLM）</td>
    <td>Llama2-7B (vLLM)</td>
  </tr>
  <tr>
    <td>MiniCPM-V-2 (vLLM）</td>
    <td>Qwen-7B (vLLM)</td>
    <td>Qwen1.5-32B-Chat (vLLM）</td>
  </tr>
  <tr>
    <td>Qwen2-72B-Instruct (vLLM）</td>
    <td>Qwen2-7B-Instruct (vLLM）</td>
    <td>StableLM2-1.6B (vLLM）</td>
  </tr>
</table>

### 问题修复

* 新增了IGIE推理模型自动化测试的运行脚本。
* 修复了YOLOv8 IxRT模型运行推理脚本报错的问题。
* 更新了YOLOv9和YOLOv10的IGIE模型的配置文件。
* 完善了IxRT模型BERT，Mask RCNN，MobileNetV2和YOLOX的end2end推理时间打印。

### 版本关联

DeepSparkInference 24.12对应天数软件栈4.1.2版本。

### 感谢以下社区贡献者

YoungPeng，majorli，xinchi.tian，xiaomei.wang，honglyua，qiang.zhang。

---

## 24.09 Release Notes

### 模型与算法

* 新增了29个推理小模型示例，其中支持IGIE推理引擎的15个，支持IxRT推理引擎的14个。

<table>
    <tr align="left">
        <th colspan=6>IGIE</th>
    </tr>
    <tr>
        <td>ATSS</td>
        <td>ConvNeXt-Small</td>
        <td>CSPDarkNet50</td>
        <td>DeiT-tiny</td>
    </tr>
    <tr>
        <td>DenseNet169</td>
        <td>EfficientNet-B2</td>
        <td>EfficientNetV2-M</td>
        <td>FCOS</td>
    </tr>
    <tr>
        <td>FSAF</td>
        <td>RepVGG</td>
        <td>RetinaFace</td>
        <td>RTMDet</td>
    </tr>
    <tr>
        <td>RTMPose</td>
        <td>SEResNet50</td>
        <td>ShuffleNet_V2_X1_0</td>
        <td></td>
    </tr>
    <tr align="left">
        <th colspan=6>IxRT</th>
    </tr>
    <tr>
        <td>ALBERT</td>
        <td>Conformer</td>
        <td>DeBERTa</td>
        <td>FaceNet</td>
    </tr>
    <tr>
        <td>RoBERTa</td>
        <td>RoFormer</td>
        <td>Swin Transformer Large</td>
        <td>Transformer ASR</td>
    </tr>
    <tr>
        <td>VideoBERT</td>
        <td>Wide_ResNet50</td>
        <td>Wide&Deep</td>
        <td>YOLOv4</td>
    </tr>
    <tr>
        <td>YOLOv6</td>
        <td>YOLOv8</td>
        <td></td>
        <td></td>
    </tr>
</table>

* 新增了5个大语言模型的推理示例。
  * Llama3-70B (TensorRT-LLM)
  * Qwen1.5-7B (Text Generation Inference)
  * Qwen1.5-7B (vLLM)
  * Qwen1.5-14B (vLLM)
  * Qwen1.5-72B (vLLM)

### 问题修复

* 修复了BERT Base SQUAD模型在NV环境上int8精度异常的问题。
* 修复了Mask RCNN模型在NV 24.04环境上运行编译报错的问题。
* 修复了CLIP IGIE模型对transformers版本有依赖的问题。
* 完善了IxRT部分模型的end2end推理时间打印。

### 版本关联

DeepSparkInference 24.09对应天数软件栈4.1.2版本。

### 感谢以下社区贡献者

YoungPeng，majorli，xinchi.tian，xiaomei.wang，wenfeng.zhang，haoyanlong，qiang.zhang。

---

## 24.06 Release Notes

### 模型与算法

* 新增了33个推理小模型示例，其中支持IGIE推理引擎的16个，支持IxRT推理引擎的15个。

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

YoungPeng，majorli，xinchi.tian，tianxi-yi，may，xiaomei.wang，cheneychen2023，qiang.zhang。

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
