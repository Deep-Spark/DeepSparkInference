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

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.4.0 | 26.06 |

**Note**: Requires at least 48GB GPU memory for single-card deployment.

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