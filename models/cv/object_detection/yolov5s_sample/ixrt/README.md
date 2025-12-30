# 此代码是检测网络基于coco数据集的通用实现


## 推理流程(以Yolov5s为例进行说明)
在ixrt-modelzoo/executables/yolov5s路径下

1. 下载onnx文件、数据集  && 安装依赖包
```
bash init.sh
```

2. 执行脚本(所需的量化、build engine等步骤都包含)

```
bash infer_yolov5s_int8_accuracy.sh --bs 32 --tgt 0.55
bash infer_yolov5s_int8_performance.sh --bs 32 --tgt 2000
```


## 如何添加新模型
1. 添加模型相关配置
ixrt-modelzoo/benchmarks/cv/detection/general_impl/trt/config/{MODELNAME_CONFIG}
2. 添加执行脚本
+ ixrt-modelzoo/executables/{model_name}/init.sh
+ ixrt-modelzoo/executables/{model_name}/infer_{model_name}_{precision}_{task}.sh