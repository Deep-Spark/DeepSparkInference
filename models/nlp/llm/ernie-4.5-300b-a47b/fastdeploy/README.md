# ERNIE-4.5-300B-A47B (FastDeploy)

## Model Description

ERNIE-4.5-300B-A47B is a state-of-the-art large-scale language model developed by Baidu and released in June 2025 under
the Apache 2.0 license. It employs a heterogeneous MoE architecture with 300B total parameters and 47B activated
parameters per token, enabling efficient multimodal understanding while maintaining strong text performance. Trained on
PaddlePaddle, it achieves 47% Model FLOPs Utilization (MFU) during pre-training through FP8 mixed precision and
fine-grained recomputation.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| BI-V150 | 4.3.0     |  25.09  |

Currently, the entire model needs to be loaded into the host memory, which requires more than 600GB of host memory. This
issue will be optimized in subsequent versions.

## Model Preparation

### Prepare Resources

- Model: <https://huggingface.co/baidu/ERNIE-4.5-300B-A47B-Paddle>



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
