<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-heading -->
# Releasing DeepSparkInference

## Release Versioning

本项目采用基于发布年月的版本号命名策略，格式为 YY.MM，发布节奏为按季度发布，一般在每年的 3/6/9/12 月发布正式版本，版本号对应为 YY.03/YY.06/YY.09/YY.12。已发布版本与 IXUCA SDK 关联关系如下表所示：

| Release Date | Release Version | IXUCA SDK |
|--------------|-----------------|-----------|
| Dec 2025     | 25.12           | v4.3.0    |
| Sep 2025     | 25.09           | v4.3.0    |
| Jun 2025     | 25.06           | v4.2.0    |
| Mar 2025     | 25.03           | v4.2.0    |
| Dec 2024     | 24.12           | v4.1.2    |
| Sep 2024     | 24.09           | v4.1.2    |
| Jun 2024     | 24.06           | v4.0.0    |
| Mar 2024     | 24.03           | v4.0.0    |

## Release Notes

### DeepSparkInference 25.12

#### 模型与算法

* 新增了 15 个推理小模型示例，其中支持 IGIE 推理引擎的 9 个，支持 ixRT 推理引擎的 6 个。
* 新增了 8 个大语言模型推理示例，其中 7 个使用 [vLLM](https://github.com/vllm-project/vllm)，1 个使用 Diffusers。

<table>
  <th colspan=3>ixRT</th>
  <tr>
      <td>DeepSort</td>
      <td>FastReID</td>
      <td>Transformer</td>
  </tr>
  <tr>
      <td>YOLOF </td>
      <td>YOLOv12</td>
      <td>YOLOv13</td>
  </tr>
  <th colspan=3>IGIE</th>
  <tr>
      <td>EfficientNet-B7</td>
      <td>FreeAnchor</td>
      <td>RegNet_X_800MF </td>
  </tr>
  <tr>
      <td>RegNet_X_8GF</td>
      <td>PISA</td>
      <td>YOLOv8-N </td>
  </tr>
  <tr>
    <td>YOLOv9</td>
    <td>YOLOv10</td>
    <td>YOLOv11</td>
  </tr>
  <th colspan=3>LLM</th>
  <tr>
      <td>NVLM-D (vLLM)</td>
      <td>PaliGemma (vLLM)</td>
      <td>Phi-3 Vision (vLLM)</td>
  </tr>
  <tr>
      <td>Pixtral (vLLM)</td>
      <td>Qwen3Moe (vLLM)</td>
      <td>Stable Diffusion 3 (Diffusers)</td>
  </tr>
  <tr>
      <td>Step3-VL (vLLM)</td>
      <td>XLM-RoBERTa (vLLM)</td>
      <td></td>
</table>

#### 修复更新

* 新增了对模型推理的 batchsize 参数配置的支持，并在 CI 中添加相应测试 (#ID8SDF, #IDBGCP)
* 新增了 vLLM 推理模型的 benchmark 脚本支持 (#ID8TTL)
* 优化了 21 个推理小模型的 README 指导文档说明 (#IDBBZV)
* 更新了 Conformer IGIE 和 BERT Large SQuAD ixRT 模型失效的链接 (#ID8KFK, #ID9RDW)
* 修复了 CosyVoice2-0.5B 模型推理报错的问题 (#ID5Y84)
* 修复了 Fuyu-8B 模型推理报错的问题 (#ID5Y8O)
* 修复了 YOLOv8 ixRT 模型推理日志中的警告打印问题 (#IDC1OI)
* 修复了 Conformer 模型解读异常问题 (#ID9RDQ)
* 修复了 EfficientNet-B1 和 YOLOv4 模型 int8 推理精度异常的问题 (#ID912Z)
* 修复了 ResNetV1d-50 和 RetinaFace ixRT 模型推理报错的问题 (#ID931D, ID94XO)
* 修复了 YOLO 系列模型在在 PyTorch 2.7 环境导出 ONNX 时的兼容报错问题 (#ID95LK)
* 修复了 8 个推理小模型在 CI 上运行报错的问题 (#ID9DH4)
* 修复了 YOLOv4 ixRT 模型量化 ONNX 报错的问题 (#IDA3BX)
* 修复了 YOLOv10 ixRT 模型 dynamic shape 的问题（#IDAOW0）

#### 版本关联

DeepSparkInference 25.12 对应天数软件栈 4.3.0 版本。

#### 感谢以下社区贡献者

YoungPeng，anders，fhfang，郭寒冰，qiang.zhang，majorli6，honglyua。

### DeepSparkInference 25.09

#### 模型与算法

* 新增了 19 个推理小模型示例，其中支持 IGIE 推理引擎的 12 个，支持 ixRT 推理引擎的 7 个。
* 新增了 11 个大语言模型推理示例，涉及 [FastDeploy](https://github.com/PaddlePaddle/FastDeploy)、[LMDeploy](https://github.com/InternLM/lmdeploy) 和 [vLLM](https://github.com/vllm-project/vllm) 等框架。

<table>
  <th colspan=3>ixRT</th>
  <tr>
    <td>ATSS</td>
    <td>CLIP</td>
    <td>CSPResNeXt50</td>
  </tr>
  <tr>
    <td>EfficientNet-B4</td>
    <td>EfficientNet-B5</td>
    <td>PAA</td>
  </tr>
  <tr>
    <td>RetinaNet</td>
    <td></td>
    <td></td>
  </tr>
  <th colspan=3>IGIE</th>
  <tr>
    <td>EfficientNet-B6</td>
    <td>MNASNet1_3</td>
    <td>Regnet_x_32gf</td>
  </tr>
  <tr>
    <td>Regnet_x_400mf</td>
    <td>Regnet_y_3_2gf</td>
    <td>Regnet_y_32gf</td>
  </tr>
  <tr>
    <td>RegNet_y_400mf</td>
    <td>SSD</td>
    <td>VGG13</td>
  </tr>
  <tr>
    <td>VGG13_BN</td>
    <td>YOLOF</td>
    <td>YOLOv13</td>
  </tr>
  <th colspan=3>大模型</th>
  <tr>
    <td>CosyVoice2-0.5B</td>
    <td>E5-V (vLLM)</td>
    <td>ERNIE-4.5-21B-A3B (FastDeploy)</td>
  </tr>
  <tr>
    <td>ERNIE-4.5-300B-A47B (FastDeploy)</td>
    <td>GLM-4V (vLLM)</td>
    <td>InternLM3 (LMDeploy)</td>
  </tr>
  <tr>
    <td>MiniCPM-o-2_6 (vLLM)</td>
    <td>Qwen-VL (vLLM)</td>
    <td>Qwen2-VL (vLLM)</td>
  </tr>
  <tr>
    <td>Qwen2.5-VL (vLLM)</td>
    <td>Whisper (vLLM)</td>
    <td></td>
  </tr>
</table>

#### 修复更新

* 修复了 BERT Large SQuAD 模型链接 404 问题 (#ICSF66)
* 修复了 4.3.0 容器环境下部分模型的依赖缺失问题
* 修复了 MViTv2-base 模型推理异常的问题
* 更新了 CLIP 模型的最新代码
* 更新了 ByteMLPerf 工具箱中 optimizer 的代码出处 (#ICKHTC)
* 更新了 4.3.0 适用大/小模型推理的 Docker 使用说明 (#ICLDBK)

#### 版本关联

DeepSparkInference 25.09 对应天数软件栈 4.3.0 版本。

#### 感谢以下社区贡献者

YoungPeng，fhfang，郭寒冰，qiang.zhang，majorli6，honglyua。

### DeepSparkInference 25.06

#### 模型与算法

* 新增了 24 个推理小模型示例，其中支持 IGIE 推理引擎的 15 个，支持 IxRT 推理引擎的 9 个。
* 新增了 6 个基于 vLLM 的大语言模型推理示例，其中 3 个为多模态模型。

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

#### 修复更新

* 修复了模型代码中的绝对路径的问题。
* 修复了 EfficientNetV2 模型依赖 timm 版本的问题。
* 修复了 GoogleNet 和 InceptionV3 模型（IGIE）的编译报错问题。
* 更新了 stable-diffusion-v1-5 模型在 huggingface 上的链接。
* 更新了 MiniCPM-V2 推理模型代码示例。
* 优化了 IxRT 和 IGIE 中视觉分类模型的公共代码避免重复。
* 增加了 DeepSparkInference 全部模型的 json 配置文件。
* 增加了模型库首页的英文版 README 文档。

#### 版本关联

DeepSparkInference 25.06 对应天数软件栈 4.2.0 版本。

#### 感谢以下社区贡献者

YoungPeng，majorli6，honglyua，qiang.zhang。

### DeepSparkInference 25.03

#### 模型与算法

* 新增了 25 个推理小模型示例，其中支持 IGIE 推理引擎的 15 个，支持 IxRT 推理引擎的 10 个。
* 新增了 11 个大模型推理示例，其中 6 个为 DeepSeek-R1 蒸馏模型。

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

#### 问题修复

* 新增了 IxRT 的 NLP 推理模型的自动化测试运行脚本。
* 优化了 IGIE 推理模型自动化测试的运行脚本。
* 修复了 onnxruntime 1.17.1 导致的 quantize fail 问题。
* 修复了 BERT 模型在 INT8 精度下的问题。
* 修复了 YOLOv6 IGIE 模型运行推理脚本报错的问题。
* 修复了 mmpretraino 0.24.0 与 mmcv 2.1.0 版本不兼容问题。
* 修复了 ChatGLM3-6B-32k 模型中的中文乱码问题。
* 修复了 vLLM 模型中 SamplingParams 的初始化问题。
* 更新了所有模型 README 文档格式，补充了模型所支持的 IXUCA SDK 版本。

#### 版本关联

DeepSparkInference 25.03 对应天数软件栈 4.2.0 版本。

#### 感谢以下社区贡献者

YoungPeng，majorli6，xinchi.tian，xiaomei.wang，honglyua，qiang.zhang。

### DeepSparkInference 24.12

#### 模型与算法

* 新增了 24 个推理小模型示例，其中支持 IGIE 推理引擎的 15 个，支持 IxRT 推理引擎的 9 个。
* 新增了 9 个大语言模型的推理示例，其中支持 vLLM 的 8 个，支持 ixFormer 的 1 个。

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

#### 问题修复

* 新增了 IGIE 推理模型自动化测试的运行脚本。
* 修复了 YOLOv8 IxRT 模型运行推理脚本报错的问题。
* 更新了 YOLOv9 和 YOLOv10 的 IGIE 模型的配置文件。
* 完善了 IxRT 模型 BERT，Mask RCNN，MobileNetV2 和 YOLOX 的 end2end 推理时间打印。

#### 版本关联

DeepSparkInference 24.12 对应天数软件栈 4.1.2 版本。

#### 感谢以下社区贡献者

YoungPeng，majorli，xinchi.tian，xiaomei.wang，honglyua，qiang.zhang。

---

### DeepSparkInference 24.09

#### 模型与算法

* 新增了 29 个推理小模型示例，其中支持 IGIE 推理引擎的 15 个，支持 IxRT 推理引擎的 14 个。

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

* 新增了 5 个大语言模型的推理示例。
  * Llama3-70B (TensorRT-LLM)
  * Qwen1.5-7B (Text Generation Inference)
  * Qwen1.5-7B (vLLM)
  * Qwen1.5-14B (vLLM)
  * Qwen1.5-72B (vLLM)

#### 问题修复

* 修复了 BERT Base SQUAD 模型在 NV 环境上 int8 精度异常的问题。
* 修复了 Mask RCNN 模型在 NV 24.04 环境上运行编译报错的问题。
* 修复了 CLIP IGIE 模型对 transformers 版本有依赖的问题。
* 完善了 IxRT 部分模型的 end2end 推理时间打印。

#### 版本关联

DeepSparkInference 24.09 对应天数软件栈 4.1.2 版本。

#### 感谢以下社区贡献者

YoungPeng，majorli，xinchi.tian，xiaomei.wang，wenfeng.zhang，haoyanlong，qiang.zhang。

---

### DeepSparkInference 24.06

#### 模型与算法

* 新增了 33 个推理小模型示例，其中支持 IGIE 推理引擎的 16 个，支持 IxRT 推理引擎的 15 个。

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

* 新增了 4 个大语言模型的推理示例。
  * Baichuan2-7B (vLLM)
  * ChatGLM-3-6B (vLLM)
  * Llama2-7B (TensorRT-LLM)
  * Qwen-7B (Text Generation Inference)

#### 问题修复

* 修复了 YOLOX IxRT 插件编译会报错的问题。
* 完善了 libGL 依赖在 Ubuntu 上安装的帮助说明。

#### 版本关联

DeepSparkInference 24.06 对应天数软件栈 4.0.0 版本。

#### 感谢以下社区贡献者

YoungPeng，majorli，xinchi.tian，tianxi-yi，may，xiaomei.wang，cheneychen2023，qiang.zhang。

---

### DeepSparkInference 24.03

#### 模型与算法

* 新增了 48 个推理模型示例，其中支持 IGIE 推理引擎的 28 个，支持 IxRT 推理引擎的 20 个。

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

#### 版本关联

DeepSparkInference 24.03 对应天数软件栈 4.0.0 版本。
