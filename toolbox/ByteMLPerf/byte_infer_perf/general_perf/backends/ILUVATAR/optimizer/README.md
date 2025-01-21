# IxRT optimizer

## 1. optimizer 简介

`optimizer` 是一个 ixrt 中集成的图融合工具，用于将onnx图中的op融合成对应的IxRT plugin，一般与 IxRT 配合进行使用；

## 2. optimizer 功能说明

| 功能       | 说明                                                         |
| ---------- | ------------------------------------------------------------ |
| 动态图支持 | 支持融合动态图和静态图                                       |
| 模型支持   | 目前测试通过videobert, roberta, deberta, swinL, roformer, albert, yolov5s, visionTransformer, gpt2模型，其他模型暂不推荐使用该工具 |

## 3. optimizer 运行参数

| 参数             | 说明                                                         |
| ---------------- | ------------------------------------------------------------ |
| `--onnx`         | 必选 ，指定要运行的 onnx 模型路径                            |
| `--num_heads`    | 可选 ，指定模型对应Attention模块注意力头的个数               |
| `--hidden_size`  | 可选， 模型模型隐藏层的大小                                  |
| `--input_shapes` | 可选 ，固定动态模型的输入形状，以从静态形状推理，示例 --input_shapes "input_name1:3x224x224, input_name2:3x224x224"类型 |
| `--dump_onnx`    | 可选 ，用于图融合过程中dump出中间的onnx图，生成 _sim 结尾的 onnx 模型 |
| `--model_type`   | 可选 ，可以指定要融合的模型类型，默认是"bert", 可选["bert", "swint", "roformer", "yolo", "gpt2", "vit"] |
| `--log_level`    | 可选 ，指定IxRT运行时显示日志的等级， 可指定为debug、info、error，默认为 info |


## 4. 运行示例

###  4.1 示例1：融合albert|videobert|roberta|deberta

```bash
cd oss/tools/optimizer
python3 optimizer.py --onnx ${MODEL_PATH}
```

###  4.2 示例2：融合swinL

```bash
cd oss/tools/optimizer
python3 optimizer.py --onnx ${MODEL_PATH} --input_shapes pixel_values.1:${BS}x3x384x384 --model_type swint
```

###  4.3 示例3：融合roformer

```bash
cd oss/tools/optimizer
python3 optimizer.py --onnx ${MODEL_PATH} --model_type roformer
```

###  4.4 示例4：融合yolov5s

```bash
cd oss/tools/optimizer
python3 optimizer.py --onnx ${MODEL_PATH} --model_type yolo
```

### 4.5 精度验证

#### 4.5.1 示例1：albert模型

模型变量示例：

```
MODEL_PATH="data/albert/albert-base-squad.onnx"
MODEL_END_PATH="data/albert/albert-base-squad_end.onnx"
MODEL_ENGINE_PATH="data/albert/albert-base-squad_end.engine"
```

运行命令

```bash
cd oss/tools/optimizer
python3 optimizer.py --onnx ${MODEL_PATH} --dump_onnx
ixrtexec --onnx ${MODEL_END_PATH} --min_shape input_ids.1:${BS}x384,attention_mask.1:${BS}x384,token_type_ids.1:${BS}x384 \
                                  --opt_shape input_ids.1:${BS}x384,attention_mask.1:${BS}x384,token_type_ids.1:${BS}x384 \
                                  --max_shape input_ids.1:${BS}x384,attention_mask.1:${BS}x384,token_type_ids.1:${BS}x384 \
                                  --save_engine ${MODEL_ENGINE_PATH} --log_level verbose --plugins ixrt_plugin
ixrtexec --load_engine ${MODEL_ENGINE_PATH} --ort_onnx ${MODEL_PATH} --plugins ixrt_plugin --verify_acc
```

#### 4.5.2 示例2：swinL模型

模型变量示例：

```
BS=1
MODEL_PATH="data/swint/swin-transformer-large.onnx"
MODEL_END_PATH = "data/swint/swin-transformer-large_end.onnx"
MODEL_ENGINE_PATH = "data/swint/swin-transformer-large_end.engine"
MODEL_SIM_STATIC_SIM_PATH = "data/swint/swin-transformer-large_sim_static_sim.onnx"
```

运行命令

```bash
cd oss/tools/optimizer
# 固定输入形状为 ${BS}x3x384x384
python3 optimizer.py --onnx ${MODEL_PATH} --input_shapes pixel_values.1:${BS}x3x384x384 --model_type swint --dump_onnx

# Build engine
ixrtexec --onnx ${MODEL_END_PATH} --save_engine ${MODEL_ENGINE_PATH} --log_level verbose --plugins ixrt_plugin

# 测试性能
ixrtexec --load_engine ${MODEL_ENGINE_PATH} --plugins ixrt_plugin

# 测试精度
ixrtexec --load_engine ${MODEL_ENGINE_PATH} --ort_onnx ${MODEL_SIM_STATIC_SIM_PATH} --plugins ixrt_plugin --verify_acc
```

请参考[高级话题](5_advanced_topics.md)中的<u>精度对比工具</u>一节，了解详细使用方法和原理。

也可以用[C++ API 使用简介](3_cpp_api.md)或 [Python API 使用简介](4_python_api.md)

具体使用方法可以参考oss/samples
