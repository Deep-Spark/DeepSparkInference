# OvisOCR2 (vLLM)

## Model Description

We are pleased to announce the release of OvisOCR2, a compact 0.8B end-to-end model for page-level document parsing. Given a document page image, OvisOCR2 generates a Markdown representation in natural reading order, covering text, formulas, tables, and visual regions.

OvisOCR2 is developed by post-training Qwen3.5-0.8B using a carefully designed data engine that combines real-world and synthetic data, together with a multi-stage training recipe integrating SFT, RL, and OPD. The model delivers strong document parsing performance while maintaining a small deployment footprint.

## Supported Environments

| GPU | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release | Branch |
| :----: | :----: | :----: | :----: |
| BI-V150 | 5.0.0 | 26.12 | release/26.12 |

> **Note:** 请切换到与您的 SDK 版本对应的 Release 分支进行测试。请勿直接在 master 分支上运行测试，因为 master 分支可能包含与您的本地 SDK 版本不兼容的最新更改。
>
> 切换分支命令示例：`git checkout release/26.12`

## Model Preparation

### Prepare Resources

- Model: <https://modelscope.cn/models/ATH-MaaS/OvisOCR2>

```bash
cp -r ../../vllm_public_assets/ ./

mkdir -p data/
ln -s /path/to/OvisOCR2 ./data/
```

`assets/document_page.png` 是随目录提供的样例文档页，两个推理脚本默认用它。
OvisOCR2 做的是文档页解析，换成风景照之类没有文字的图片时它读不到内容，会一直输出
填充文本直到 `--max-tokens` 耗尽。换图请换成文档扫描页。

### Install Dependencies

In order to run the model smoothly, you need to get the sdk from [resource
center](https://support.iluvatar.com/#/ProductLine?id=2) of Iluvatar CoreX official website.

## Online Inference (Client-Server Mode)

### Start the server (Server)

```bash
export CUDA_VISIBLE_DEVICES=1,0
vllm serve ./data/OvisOCR2 --served-model-name OvisOCR2 --port 8000 --tensor-parallel-size 2 --pipeline-parallel-size 1 --max-model-len 4096 --trust-remote-code --gpu-memory-utilization 0.9 --enforce-eager
```

Wait until the endpoint is ready:

```bash
until curl -fsS http://127.0.0.1:8000/v1/models >/dev/null; do sleep 5; done
```

> **关于 `--enforce-eager`**：本次启动显式关闭了 torch.compile 与 CUDA Graph 捕获，用于绕开推理栈图编译路径上的缺陷。
> 它不改变权重、镜像、软件栈与驱动，只改变算子的执行方式。下方吞吐是 eager 模式的结果，不代表该镜像图模式下的性能上限。

### Run the client (Client)

After the server starts up, use `online_inference_vision_language.py` to send requests.
A local image is inlined as a data URL; an `http(s)` address is passed through as is.

```bash
python3 online_inference_vision_language.py \
  --model-name "OvisOCR2" \
  --image ./assets/document_page.png \
  --prompt "Parse this document page into Markdown."
```

## Offline Inference

`offline_inference_vision_language.py` 在本进程内起引擎，不需要先启服务，适合单机跑通与回归。
它走 `LLM.chat`，由权重自带的 chat template 拼接图文，和服务端收到的 prompt 一致。

```bash
export CUDA_VISIBLE_DEVICES=1,0
python3 offline_inference_vision_language.py \
  --model ./data/OvisOCR2 \
  --image ./assets/document_page.png \
  --prompt "Parse this document page into Markdown."
```

> 张量并行会把本文件在 worker 进程里重新 import，所以引擎必须放在 `if __name__ == "__main__":`
> 之后；直接写在模块顶层会让每个 worker 再建一遍引擎并报
> `An attempt has been made to start a new process before the current process has finished its bootstrapping phase`。

## Model Results

### Benchmarking Results

| Model | Precision  | QPS | Total TPS | Output TPS |
| :----: | :----: | :----: | :----: | :----: |
| OvisOCR2 | - | 0.56 | 69.93 | 59.25 |

> 以上结果在 eager 模式下测得，共 100 个请求，总耗时 177.752 s。

### Offline Inference Results

| Model | Precision | Check | Output Tokens | Load | Latency | Result |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| OvisOCR2 | BF16 | Parse document page to Markdown | 92 | 161.6 s | 26.1 s | Pass |

> 以上结果在 2 张 BI-V150、`--tensor-parallel-size 2`、eager 模式下测得，
> 输入为 `assets/document_page.png`。输出的标题、正文与参数表与原页逐行一致。

## References

- [vLLM](https://github.com/vllm-project/vllm)
