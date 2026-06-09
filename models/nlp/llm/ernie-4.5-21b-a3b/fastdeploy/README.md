# ERNIE-4.5-21B-A3B (FastDeploy)

## Model Description

ERNIE-4.5-21B-A3B is a large-scale Mixture of Experts (MoE) language model developed by Baidu, featuring 21 billion
total parameters with 3 billion activated parameters per token. The model employs a heterogeneous MoE architecture with
64 text experts and 64 vision experts, activating 6 experts per token. It boasts an ultra-long context window of 131,072
tokens across 28 layers. The model supports multimodal heterogeneous pre-training and utilizes 4-bit/2-bit lossless
quantization for efficient inference. Built on PaddlePaddle framework, it excels in text understanding and generation
tasks, supporting dialogue, question-answering, and various other applications as a key member of the ERNIE 4.5 series.

## Supported Environments

| GPU | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release | Branch |
| :----: | :----: | :----: | :----: |
| MR-V100 | 4.4.0 | 26.03 | release/26.03 |
| MR-V100 | 4.3.0 | 25.12 | release/25.12 |

> **Note:** 请切换到与您的 SDK 版本对应的 Release 分支进行测试。请勿直接在 master 分支上运行测试，因为 master 分支可能包含与您的本地 SDK 版本不兼容的最新更改。
>
> 切换分支命令示例：`git checkout release/26.03`

## Model Preparation

### Prepare Resources

- Model: <https://huggingface.co/baidu/ERNIE-4.5-21B-A3B-Paddle>



```sh
# Pull the docker image
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-ixuca:latest

# Start Container
docker run -itd --name paddle_infer -v /usr/src:/usr/src -v /lib/modules:/lib/modules -v /dev:/dev -v /home/paddle:/home/paddle --privileged --cap-add=ALL --pid=host ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-ixuca:latest
docker exec -it paddle_infer bash
```

### Install Dependencies

```sh
pip3 install paddlepaddle==3.1.0a0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
pip3 install paddle-iluvatar-gpu==3.1.0 -i https://www.paddlepaddle.org.cn/packages/stable/ixuca/
pip3 install fastdeploy_iluvatar_gpu -i https://www.paddlepaddle.org.cn/packages/stable/ixuca/ --extra-index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

## Model Inference

```sh
chmod u+x ./run_demo.sh
./run_demo.sh
```

## References

- [FastDeploy](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/get_started/installation/iluvatar_gpu.md)
