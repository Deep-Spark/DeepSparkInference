# Grounding DINO (ixRT)

## Model Description

Grounding DINO is a state-of-the-art open-set (or zero-shot) object detection model that combines vision and language to detect objects in images based on text prompts—without requiring pre-defined, fixed categories during training.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.4.0 | 26.03 |

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

| Model | Precision | FPS    | IOU@0.5 | IOU@0.5:0.95 |
| :----: | :----: | :----: | :----: | :----: |
| Grounding DINO | FP16      | 75.199 | 0.491   | 0.327        |

## References

- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
