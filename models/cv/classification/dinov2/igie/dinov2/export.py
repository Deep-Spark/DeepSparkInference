# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""Export DINOv2 backbone to ONNX."""
import os

# Iluvatar ixformer flash-attn only supports fp16/bf16 and is not ONNX-traceable.
os.environ.setdefault("XFORMERS_DISABLED", "1")

"""
The exported graph expects **already preprocessed** ImageNet-style tensors:
  - shape: [batch, 3, H, W]  (default H=W=224, same as linear eval)
  - dtype: float32
  - value range after preprocessing:
      Resize(256) -> CenterCrop(224) -> ToTensor -> Normalize(
          mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
      )

Outputs match ``DinoVisionTransformer.get_intermediate_layers(..., norm=True,
return_class_token=True)`` for the last ``n_last_blocks`` transformer blocks:
  - cls_tokens:   [batch, n_last_blocks, embed_dim]
  - patch_tokens: [batch, n_last_blocks, num_patches, embed_dim]

This is sufficient for linear eval (n_last_blocks_list=[1, 4], avgpool in {False, True})
to produce numerically identical features as the PyTorch .pth backbone.
"""

import argparse
import logging
from pathlib import Path
from typing import Tuple

import torch
from omegaconf import OmegaConf
from torch import nn

from dinov2.configs import dinov2_default_config
from dinov2.data.transforms import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from dinov2.eval.setup import get_args_parser as get_setup_args_parser
from dinov2.logging import setup_logging
from dinov2.models import build_model_from_cfg
import dinov2.utils.utils as dinov2_utils


logger = logging.getLogger("dinov2")


class DinoV2IntermediateLayersWrapper(nn.Module):
    """Export-friendly wrapper: preprocessed image tensor -> intermediate layer tokens."""

    def __init__(self, backbone: nn.Module, n_last_blocks: int = 4):
        super().__init__()
        self.backbone = backbone
        self.n_last_blocks = n_last_blocks

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        layers = self.backbone.get_intermediate_layers(
            images,
            n=self.n_last_blocks,
            return_class_token=True,
            norm=True,
        )
        cls_tokens = torch.stack([cls for _, cls in layers], dim=1)
        patch_tokens = torch.stack([patch for patch, _ in layers], dim=1)
        return cls_tokens, patch_tokens


def load_model(
    config_file: str,
    pretrained_weights: str,
    device: str,
    n_last_blocks: int,
) -> nn.Module:
    default_cfg = OmegaConf.create(dinov2_default_config)
    cfg = OmegaConf.merge(default_cfg, OmegaConf.load(config_file))
    model, _ = build_model_from_cfg(cfg, only_teacher=True)
    dinov2_utils.load_pretrained_weights(model, pretrained_weights, "teacher")
    model = model.to(device)
    model.eval()
    return DinoV2IntermediateLayersWrapper(model, n_last_blocks=n_last_blocks)


def get_export_parser() -> argparse.ArgumentParser:
    parser = get_setup_args_parser(description="Export DINOv2 backbone to ONNX")
    parser.add_argument(
        "--onnx-path",
        type=str,
        default="",
        help="Output ONNX path (default: <pretrained-weights>.onnx)",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=224,
        help="Spatial size H=W of the preprocessed input tensor",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Dummy batch size used for ONNX tracing",
    )
    parser.add_argument(
        "--n-last-blocks",
        type=int,
        default=4,
        help="Number of last transformer blocks to export (linear eval uses max=4)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device used for tracing/export",
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=17,
        help="ONNX opset version",
    )
    return parser


def main() -> int:
    setup_logging()
    args = get_export_parser().parse_args()

    if not args.config_file:
        raise ValueError("--config-file is required (e.g. dinov2/configs/eval/vits14_pretrain.yaml)")
    if not args.pretrained_weights:
        raise ValueError("--pretrained-weights is required (e.g. dinov2_vits14_pretrain.pth)")

    onnx_path = args.onnx_path or args.pretrained_weights.replace(".pth", ".onnx")
    onnx_path = str(Path(onnx_path))

    model = load_model(
        args.config_file,
        args.pretrained_weights,
        args.device,
        n_last_blocks=args.n_last_blocks,
    )
    dummy_input = torch.randn(
        args.batch_size,
        3,
        args.input_size,
        args.input_size,
        device=args.device,
    )

    with torch.inference_mode():
        sample_cls, sample_patch = model(dummy_input)
    logger.info("Export input shape: %s", tuple(dummy_input.shape))
    logger.info("Export cls_tokens shape: %s", tuple(sample_cls.shape))
    logger.info("Export patch_tokens shape: %s", tuple(sample_patch.shape))
    logger.info(
        "Preprocessing expected before ONNX: Resize(256)->CenterCrop(%d)->ToTensor->Normalize(mean=%s, std=%s)",
        args.input_size,
        IMAGENET_DEFAULT_MEAN,
        IMAGENET_DEFAULT_STD,
    )

    dynamic_axes = {
        "input": {0: "batch_size", 2: "height", 3: "width"},
        "cls_tokens": {0: "batch_size"},
        "patch_tokens": {0: "batch_size"},
    }
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["cls_tokens", "patch_tokens"],
        dynamic_axes=dynamic_axes,
        opset_version=args.opset_version,
        dynamo=False,
        external_data=False,
        do_constant_folding=True,
    )
    logger.info("Exported ONNX to %s", onnx_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
