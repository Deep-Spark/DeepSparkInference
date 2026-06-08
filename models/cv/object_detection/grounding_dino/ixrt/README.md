# Grounding DINO (ixRT)

## Model Description

Grounding DINO is a state-of-the-art open-set (or zero-shot) object detection model that combines vision and language to detect objects in images based on text prompts—without requiring pre-defined, fixed categories during training.

## Supported Environments

| GPU | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release | Branch |
| :----: | :----: | :----: | :----: |
| MR-V100 | dev-only | dev- | release/dev- |
| MR-V100 | 4.4.0 | 26.03 | release/26.03 |

> **Note:** 请切换到与您的 SDK 版本对应的 Release 分支进行测试。请勿直接在 master 分支上运行测试，因为 master 分支可能包含与您的本地 SDK 版本不兼容的最新更改。
>
> 切换分支命令示例：`git checkout release/26.03`

## Model Preparation

### Prepare Resources

Pretrained model: <http://files.deepspark.org.cn:880/deepspark/data/checkpoints/grounded_static_1x800x1200x128_sim_end.onnx>

```bash
pip install transformers
mkdir -p checkpoints
# download model into checkpoints
```

## Model Inference

### FP16

```bash
# Accuracy
bash scripts/infer_grounding_dino_fp16_accuracy.sh
# Performance
bash scripts/infer_grounding_dino_fp16_performance.sh
```

## Model Results

| Model | Precision | FPS    | output_logits | output_boxes | boxes_filt |
| :----: | :----: | :----: | :----: | :----: | :----: |
| Grounding DINO | FP16      |  7.474 | (1, 900, 256) 921600   | (1, 900, 4) 14400  | torch.Size([8, 4]) |

## References

- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
