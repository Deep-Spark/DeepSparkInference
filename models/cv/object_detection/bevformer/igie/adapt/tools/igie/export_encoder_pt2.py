#!/usr/bin/env python3
"""Export BEVFormer encoder-only subgraph to .pt2 for IGIE."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.export import export

IGIE_DIR = Path(__file__).resolve().parent
if str(IGIE_DIR) not in sys.path:
    sys.path.insert(0, str(IGIE_DIR))

from export_utils import (  # noqa: E402
    EncoderWrapper,
    build_bevformer_model,
    ensure_repo_on_path,
    fpn_feat_shapes_from_image,
    make_head_dummy_inputs,
    patch_decoder_for_pt2_export,
    patch_head_bbox_for_pt2_export,
    patch_model_for_pt2_export,
    remove_runtime_assertions,
)
from sca_export_core import install_sca_export_patch  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description='Export BEVFormer encoder to .pt2')
    p.add_argument('config', help='config file path')
    p.add_argument('checkpoint', help='checkpoint file path')
    p.add_argument('--output', default='bevformer_encoder.pt2')
    p.add_argument(
        '--image-shape',
        type=int,
        nargs=5,
        default=[1, 6, 3, 928, 1600],
        metavar=('B', 'N', 'C', 'H', 'W'))
    p.add_argument('--device', default='cuda')
    return p.parse_args()


def _set_level_lengths(model, image_shape, device):
    from projects.mmdet3d_plugin.bevformer.modules.spatial_cross_attention import \
        MSDeformableAttention3D
    from export_utils import infer_fpn_feat_shapes

    feat_shapes = infer_fpn_feat_shapes(model, tuple(image_shape), device)
    level_lengths = tuple(int(h * w) for _, _, h, w in feat_shapes)
    for mod in model.modules():
        if isinstance(mod, MSDeformableAttention3D):
            mod._level_lengths = level_lengths
            print(f'  set _level_lengths={level_lengths} on {type(mod).__name__}')


def main():
    args = parse_args()
    ensure_repo_on_path()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    install_sca_export_patch()

    model, _ = build_bevformer_model(args.config, args.checkpoint, device)
    model.use_grid_mask = False
    patch_model_for_pt2_export(model)
    patch_decoder_for_pt2_export(model)
    patch_head_bbox_for_pt2_export(model)

    batch_size, num_cams, _, img_h, img_w = args.image_shape
    _set_level_lengths(model, args.image_shape, device)

    feat_shapes = fpn_feat_shapes_from_image(batch_size, num_cams, img_h, img_w)
    example = make_head_dummy_inputs(
        batch_size, num_cams, img_h, img_w, feat_shapes, device)
    wrapper = EncoderWrapper(model, batch_size, num_cams).eval().to(device)

    with torch.no_grad():
        ep = export(wrapper, example)
        ep = remove_runtime_assertions(ep)
        torch.export.save(ep, args.output)
        bev_embed = wrapper(*example)

    print(f'saved {args.output}')
    print(f'  bev_embed: {tuple(bev_embed.shape)}')


if __name__ == '__main__':
    main()
