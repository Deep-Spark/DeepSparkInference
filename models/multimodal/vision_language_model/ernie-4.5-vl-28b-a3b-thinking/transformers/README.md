# ERNIE-4.5-VL-28B-A3B-Thinking (Transformers)

## Model Description

ERNIE-4.5-VL-28B-A3B-Thinking is a breakthrough multimodal AI model featuring 28B total parameters with only 3B activated parameters per token. Built upon ERNIE-4.5-VL architecture with extensive multimodal reinforcement learning.

Key features:
- **Visual Reasoning**: Exceptional multi-step reasoning, chart analysis, and causal reasoning
- **STEM Reasoning**: Powerful performance on STEM tasks from photos
- **Visual Grounding**: Precise grounding and flexible instruction execution
- **Thinking with Images**: Freely zoom in/out of images to process fine-grained details
- **Tool Utilization**: Robust tool-calling capabilities for comprehensive information retrieval
- **Video Understanding**: Outstanding temporal awareness and event localization

## Supported Environments

| GPU | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release | Branch |
| :----: | :----: | :----: | :----: |
| MR-V100 | 4.4.0 | 26.06 | release/26.06 |

> **Note:** 请切换到与您的 SDK 版本对应的 Release 分支进行测试。请勿直接在 master 分支上运行测试，因为 master 分支可能包含与您的本地 SDK 版本不兼容的最新更改。
>
> 切换分支命令示例：`git checkout release/26.06`

## Model Preparation

### Prepare Resources

- Model: <https://huggingface.co/baidu/ERNIE-4.5-VL-28B-A3B-Thinking>

### Install Dependencies

```bash
pip install decord
```

## Model Inference

```bash
python3 run_demo.py
```