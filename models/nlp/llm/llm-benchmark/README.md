# 安装

```bash
# pip安装依赖
pip3 install -r requirements.txt
```

# 精度评测

## 简单评测

假如使用SGLang拉起模型服务，IP为：127.0.0.1，PORT为30000，在指定的若干数据集上使用默认配置评测DeepSeek模型，在任意路径下，执行`eval`命令：
```bash
./iluvatar_bench eval \
 --model /data/DeepSeek-R1-AWQ \
 --datasets gsm8k \
 --limit 4 \
 --eval-batch-size 8
```

### 基本参数说明

- `--model`: 指定了模型在ModelScope中的model_id，可自动下载，也可使用模型的本地路径，例如/path/to/model。
- `--datasets`: 数据集名称，支持输入多个数据集，使用空格分开，数据集将自动从modelscope下载，支持的数据集参考数[据集列表](https://evalscope.readthedocs.io/zh-cn/latest/get_started/supported_dataset/index.html)。
- `--limit`: 每个数据集子集，最大评测数据量，不填写则默认为全部评测，可用于快速验证。
- `--eval-batch-size`: 评测批量大小，默认为1，表示并发请求数量。


## 模型API服务评测

指定模型API服务地址(api_url)和API Key(api_key)，评测部署的模型API服务，此时eval-type参数指定为server，默认参数如下：
```bash
--api-key='EMPTY' \
--api-url='http://127.0.0.1:30000/v1' \
--eval-type='server'
```

# 性能测试-1

## 基本使用

下面展示了用 SGLang 框架在 Bi150 上进行 DeepSeek-R1-AWQ 模型的压测示例，固定输入1024 token，输出1024 token。用户可以根据自己的需求修改参数。

```bash
./iluvatar_bench perf \
    --parallel 1 10 50 100 200 \
    --number 10 20 100 200 400 \
    --model /data/DeepSeek-R1-AWQ \
    --url http://127.0.0.1:30000/v1/completions \
    --api openai \
    --dataset random \
    --max-tokens 1024 \
    --min-tokens 1024 \
    --prefix-length 0 \
    --min-prompt-length 1024 \
    --max-prompt-length 1024 \
    --tokenizer-path /data/DeepSeek-R1-AWQ \
    --extra-args '{"ignore_eos": true}'
```

### 参数说明

- `parallel`: 请求的并发数，可以传入多个值，用空格隔开。
- `number`: 发出的请求的总数量，可以传入多个值，用空格隔开（与`parallel`一一对应）。
- `url`: 请求的URL地址。
- `model`: 使用的模型名称。
- `api`: 使用的API服务，默认为`openai`。
- `dataset`: 数据集名称，此处为`random`，表示随机生成数据集，具体使用说明参考；更多可用的(多模态)数据集请参考数据集配置。
- `tokenizer-path`: 模型的tokenizer路径，用于计算token数量（在random数据集中是必须的）。
- `extra-args`: 请求中的额外的参数，传入json格式的字符串，例如`{"ignore_eos": true}`表示忽略结束token。

**默认参数**

其中，下列参数，属于默认参数。
```bash
--max-tokens=1024,
--min-tokens=1024,
--min-prompt-length=1024,
--max-prompt-length=1024,
--api='openai',
--url='http://127.0.0.1:30000/v1/completions'
```

`max-tokens`和`min-tokens`表示最大生成长度和最小生成长度。
`max-prompt-length`和`min-prompt-length`，表示最大提示词长度和最小提示词长度。

# 性能测试-2

这里使用SGLang自带的 bench_serving.py（源自vLLM），用于测量在线服务吞吐量和延迟。

bench_serving.py 文件信息：
```bash
git log --oneline -- ./python/sglang/bench_serving.py
88a6f9dab bench_serving support PD Disaggregation (#11542)
```

## 基本使用

注：`./iluvatar_bench sgl-perf` 命令，与 `python3 bench_serving.py` 等同，都可以运行。

```bash
./iluvatar_bench sgl-perf \
    --backend sglang \
    --host 127.0.0.1 --port 30000 \
    --num-prompts 1000
```
模型名或路径，若未设置，系统将向/v1/models请求默认模型配置。

## 公共参数

* `--backend backend`: sglang/vllm等后端。
* `--model`: 模型名称或者地址。
* 连接参数：`--host`和`--port`、或`--base-url`。
* `--dataset-name`: sharegpt, random, random-ids, generated-shared-prefix等，不同数据集，相关的配置参数不同。
* `--request-rate`：每秒到达的请求数（默认值 inf，表示所有请求同时到达），使用**泊松过程（Poisson process）**来模拟请求的到达时间。这意味着请求之间的时间间隔是随机的，但平均速率符合设定的值。这更真实地模拟了现实世界中随机到达的用户请求。假如每隔3.5秒，需要发送6条数据，则 `Request rate = 6 requests / 3.5 seconds ≈ 1.71 requests/second`。
* `--request-interval`: 固定间隔时间（秒）。如果设置，此值将覆盖 `--request-rate` 的设置，并使用确定性（固定时间）的间隔调度。
* `--max-concurrency`：最大并发请求数。表示实际处理请求的 worker 数量，虽然 `--request-rate` 参数控制请求发起的速率，但此参数控制实际允许同时执行的请求数量。。
* `--warmup-requests`: benchmark 前的 warmup 次数。

## sharegpt 数据集

`sharegpt`是真实对话数据集（默认），相关的参数如下：
* `--num-prompts`: 请求总数。
* `--sharegpt-output-len`: 输出长度，如果没有指定，则由数据集中的样本长度决定。
* `--sharegpt-context-len`: 设置上下文总体长度，被指定时，当 `输入 + 输出 > 最大上下文长度`时，request会被跳过。

简单来说，输入长度不可以被指定，输出长度和最大上下文长度，可以被指定。

注意，当出现以下情况，request同样会被跳过：
* `prompt_len < 2` 或者 `output_len < 2`

## random/random-ids数据集

* `random`: **真实的文本**，来自 ShareGPT 数据集。确定一个随机的目标输入长度（例如 500 token），它会从 ShareGPT 数据集中随机选择一个真实的提示。如果提示太长（例如 1000 token），它会截断 (truncate) 提示到 500 token；如果提示太短（例如 100 token），它会重复 (repeat) 这个提示的 token，直到填满 500 token。用于模拟一个提示内容是**真实自然语言**的随机长度工作负载。
* `random-ids`: **完全随机的 Token ID**。首先确定一个随机的目标输入长度（例如 500 token），它不会加载任何外部数据集，它会直接在 tokenizer 的词汇表（vocab）范围内随机生成 500 个 token ID。这些 ID 组合起来的文本**不具有任何语言学意义**（即"乱码"）。模拟一个提示内容是随机、无意义数据的随机长度工作负载，对于压力测试 tokenizer 和模型处理异常输入的能力很有用。

相关的参数如下：
* `--num-prompts`：要处理的请求总数
* `--random-input-len`(default:1024): 每个请求的最大输入 token 长度。脚本会在 `[random-input-len * random-range-ratio, random-input-len + 1)`随机采样一个长度。
* `--random-output-len`(default:1024): 每个请求的最大输出 token 长度。脚本会在 `[random-output-len * random-range-ratio, random-output-len + 1)`随机采样一个长度。
* `--random-range-ratio`(default:0.0): 一个介于 0.0 和 1.0 之间的浮点数，用于定义随机长度的下限。如果希望输入/输出长度固定为 1024，设置为 1.0 即可。
* `--tokenize-prompt`: 主要用于 `random` 和 `random-ids` 数据集，以便在使用 `sglang` 和 `vllm` 后端时，通过发送精确长度的 token ID 列表来进行基准测试。例如，客户端生成 `[1024个ID]` 列表，跳过解码步骤，直接将这个整数ID列表发送给服务器。服务器收到ID列表后，会跳过分词步骤，直接使用这个列表。好处就是，这保证了服务器处理的输入长度**精确地**是我们想要的1024个 token。

以下是对于输入/输出长度，如何进行最大值/最小值的计算，如果不需要，跳过以下“计算公式”和“举例说明”即可。

计算公式：
* `实际输入长度=[random-input-len * random-range-ratio, random-input-len + 1)`
* `实际输出长度=[random-output-len * random-range-ratio, random-output-len + 1)`

举例说明：
```bash
--dataset-name random \
--random-input-len 1024 \
--random-output-len 1024 \
--random-range-ratio 0.8
```
则，输入/输出长度大小，会从区间`[819, 1025) = [1024 * 0.8, 1024+1)`进行随机取值，这时候的输入/输出长度，可能是`833/955`.
如果希望输入/输出长度固定为 1024，则需要把 `--random-range-ratio` 设置为 1.0.

## generated-shared-prefix 数据集

`generated-shared-prefix` 数据集并不是一个像 `sharegpt` 那样从外部文件加载的静态数据集，而是一个动态生成的数据集。

它的核心目的是模拟一个非常重要且常见的 LLM 服务场景：大量请求共享一个长的前缀（prefix）。这通常发生在多租户、RAG（检索增强生成）或设置了复杂系统指令（system prompt）的应用中。

每个生成的请求都由两部分组成：

1. **共享的系统提示 (System Prompt)**：一个很长的、在组内共享的文本块。
2. **唯一的问题 (Question)**：一个较短的、每个请求独有的文本块。

完整的提示 (prompt) 会被构造成类似于 `"{system_prompt}\n\n{question}"` 的形式。

相关的参数如下：
* `--gsp-num-groups` (default: 64): 定义了要生成的**唯一系统提示 (system prompt) 的数量**。代表了基准测试中有多少个"共享前缀"的组。
* `--gsp-prompts-per-group` (default: 16): 定义了每个组（即每个系统提示）包含多少个唯一的请求（问题）。这决定了每个共享前缀被重用的次数。总的请求数量将是 `gsp-num-groups * gsp-prompts-per-group`。
* `--gsp-system-prompt-len` (default: 2048): 每个生成的 system prompt 的目标 token 长度。这用于模拟一个很长的前缀（例如，一个复杂的指令集或一个大的上下文文档）。
* `--gsp-question-len` (default: 128): 每个生成的唯一问题的目标 token 长度。这模拟了用户输入的、非共享的那部分提示。
* `--gsp-output-len` (default: 256): 基准测试中，为每个请求设置的目标输出 token 数量。这定义了模型在接收到 `system_prompt + question` 后需要生成多少内容。

## 例子

1. `sharegpt` (模拟真实对话)

```bash
./iluvatar_bench sgl-perf \
    --backend sglang \
    --host 127.0.0.1 --port 30000 \
    --model /home/data/qwen3/Qwen3-32B \
    --dataset-name sharegpt \
    --host 127.0.0.1 --port 30000 \
    --num-prompts 1000
```

2. `random` (模拟特定长度的合成负载)

```bash
./iluvatar_bench sgl-perf \
    --backend sglang \
    --host 127.0.0.1 --port 30000 \
    --model /home/data/qwen3/Qwen3-32B \
    --dataset-name random \
    --num-prompts 1000 \
    --random-input 2048 \
    --random-output 128 \
    --random-range-ratio 0.5
```
此命令测试1000个请求。每个请求的输入长度将在 `(2048 * 0.5)` 到 `2048` 之间随机（即 1024 到 2048 token）。输出长度将在 `(128 * 0.5)` 到 `128` 之间随机（即 64 到 128 token）。提示内容是基于ShareGPT文本填充的。

3. `random-ids` (纯粹的压力测试)

最极端的压力测试。它不关心提示的语言含义。它只是生成完全随机的 Token ID 来填满指定的输入长度。
可以与 `--tokenize-prompt` 结合使用，以发送 `[1024, 512, 300, ...]` 这样的ID列表，而不是解码后的乱码字符串。这可以**100%精确地控制输入长度**，是测量纯硬件和系统吞吐量的最佳方式。

```bash
# 压力测试: 1000个请求，每个请求 *精确地* 包含1024个输入ID
# 并请求1024个输出ID
./iluvatar_bench sgl-perf \
    --backend sglang \
    --host 127.0.0.1 --port 30000 \
    --model /home/data/qwen3/Qwen3-32B \
    --dataset-name random-ids \
    --num-prompts 1000 \
    --random-input-len 1024 \
    --random-output-len 1024 \
    --random-range-ratio 1.0 \
    --tokenize-prompt
```
`--random-range-ratio 1.0` 确保输入/输出长度不会随机化，而是精确等于 1024。`--tokenize-prompt` 确保客户端发送的是 `input_ids` 列表，而不是 `text`。这个命令是在测量服务器处理“1024-in, 1024-out”请求的纯粹性能。

4. 速率控制 + 输出文件

```bash
./iluvatar_bench sgl-perf \
    --backend sglang \
    --host 127.0.0.1 --port 30000 \
    --model /home/data/qwen3/Qwen3-32B \
    --dataset-name random \
    --random-input-len 1024 --random-output-len 1024 --random-range-ratio 1.0 \
    --num-prompts 2000 \
    --request-rate 100 \
    --max-concurrency 512 \
    --output-file sglang_random.jsonl --output-details
```

5. `generated-shared-prefix` (测试 KV Cache 性能)

```bash
./iluvatar_bench sgl-perf \
    --backend sglang \
    --host 127.0.0.1 --port 30000 \
    --model /home/data/qwen3/Qwen3-32B \
    --dataset-name generated-shared-prefix \
    --gsp-num-groups 64 --gsp-prompts-per-group 16 \
    --gsp-system-prompt-len 4096 --gsp-question-len 128 --gsp-output-len 256 \
    --num-prompts 1024
```
此命令将生成 `64 * 16 = 1024` 个总请求。它会创建 64 个不同的、长度为 4096 token 的“系统提示”（共享前缀）。

然后，对于每一个系统提示，它都会生成 16 个不同的、长度为 128 token 的“问题”。服务器在处理这1024个（被打乱的）请求时，如果其KV Cache效率高，那么它处理4096-token前缀的成本应该只支付64次，而不是1024次。

## PD Disaggregation Mode性能分析

**关键参数**

* `--pd-separated`: 启动 PD 模式。
* `--profile-prefill-url`:用于性能分析的 prefill worker数量。
* `--profile-decode-url`: 用于性能分析的 decode worker数量。

`--profile-prefill-url` 并且 `--profile-decode-url` 是相互排斥的 - 只能二选一。

Start server
```bash
# set trace path
export SGLANG_TORCH_PROFILER_DIR=/root/sglang/profile_log

# start prefill and decode servers (see PD disaggregation docs for setup)
python3 -m sglang.launch_server --model-path /home/data/qwen3/Qwen3-32B --disaggregation-mode prefill
python3 -m sglang.launch_server --model-path /home/data/qwen3/Qwen3-32B --disaggregation-mode decode --port 30001 --base-gpu-id 1

# start router
python3 -m sglang_router.launch_router --pd-disaggregation --prefill http://127.0.0.1:30000 --decode http://127.0.0.1:30001 --host 0.0.0.0 --port 8000
```

Profile Prefill Workers
```bash
# send profiling request targeting prefill workers
./iluvatar_bench sgl-perf --backend sglang --model /home/data/qwen3/Qwen3-32B --num-prompts 10 --sharegpt-output-len 100 --profile --pd-separated --profile-prefill-url http://127.0.0.1:30000
```

Profile Decode Workers
```bash
# send profiling request targeting decode workers
./iluvatar_bench sgl-perf --backend sglang --model /home/data/qwen3/Qwen3-32B --num-prompts 10 --sharegpt-output-len 100 --profile --pd-separated --profile-decode-url http://127.0.0.1:30001
```

注意：
* 这两个选项都支持用于多实例设置下的多个 worker URL：
```bash
# Profile multiple prefill workers
./iluvatar_bench sgl-perf --backend sglang --model /home/data/qwen3/Qwen3-32B --num-prompts 10 --profile --pd-separated --profile-prefill-url http://127.0.0.1:30000 http://127.0.0.1:30002

# Profile multiple decode workers
./iluvatar_bench sgl-perf --backend sglang --model /home/data/qwen3/Qwen3-32B --num-prompts 10 --profile --pd-separated --profile-decode-url http://127.0.0.1:30001 http://127.0.0.1:30003
```

# References

- [evalscope](https://github.com/modelscope/evalscope)