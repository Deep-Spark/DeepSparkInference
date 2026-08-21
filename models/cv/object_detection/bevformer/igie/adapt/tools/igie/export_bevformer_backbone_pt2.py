#!/usr/bin/env python3
"""Export BEVFormer backbone+neck to .pt2 for IGIE."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.export import export

IGIE_DIR = Path(__file__).resolve().parent
if str(IGIE_DIR) not in sys.path:
    sys.path.insert(0, str(IGIE_DIR))

from dcn_export_core import install_dcn_export_fallback  # noqa: E402
from export_utils import (  # noqa: E402
    BackboneNeckWrapper,
    build_bevformer_model,
    count_batch_norm_ops,
    ensure_repo_on_path,
    install_frozen_bn_export,
    remove_runtime_assertions,
)


def parse_args():
    p = argparse.ArgumentParser(description='Export BEVFormer backbone+neck to .pt2')
    p.add_argument('config', help='config file path')
    p.add_argument('checkpoint', help='checkpoint file path')
    p.add_argument('--output', default='bevformer_backbone.pt2')
    p.add_argument(
        '--shape',
        type=int,
        nargs=5,
        default=[1, 6, 3, 928, 1600],
        metavar=('B', 'N', 'C', 'H', 'W'))
    p.add_argument('--device', default='cuda')
    return p.parse_args()


def main():
    args = parse_args()
    ensure_repo_on_path()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    install_dcn_export_fallback()
    install_frozen_bn_export()

    model, _ = build_bevformer_model(args.config, args.checkpoint, device)
    model.use_grid_mask = False

    batch_size, num_cams, channels, img_h, img_w = args.shape
    img = torch.randn(
        batch_size, num_cams, channels, img_h, img_w,
        dtype=torch.float32, device=device)
    wrapper = BackboneNeckWrapper(model).eval().to(device)

    with torch.no_grad():
        ep = export(wrapper, (img,))
        ep = remove_runtime_assertions(ep)
        bn_ops = count_batch_norm_ops(ep)
        if bn_ops:
            print(f'WARNING: pt2 still has {bn_ops} batch_norm ops')
        torch.export.save(ep, args.output)
        feats = wrapper(img)

    print(f'saved {args.output}')
    for i, feat in enumerate(feats):
        print(f'  feat{i}: {tuple(feat.shape)}')


if __name__ == '__main__':
    main()
