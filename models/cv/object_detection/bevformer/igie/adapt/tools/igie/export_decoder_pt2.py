#!/usr/bin/env python3
"""Export BEVFormer decoder+cls/bbox heads to .pt2 for IGIE."""

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
    DecoderWrapper,
    build_bevformer_model,
    ensure_repo_on_path,
    patch_decoder_for_pt2_export,
    patch_head_bbox_for_pt2_export,
    remove_runtime_assertions,
)


def parse_args():
    p = argparse.ArgumentParser(description='Export BEVFormer decoder to .pt2')
    p.add_argument('config', help='config file path')
    p.add_argument('checkpoint', help='checkpoint file path')
    p.add_argument('--output', default='bevformer_decoder.pt2')
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--device', default='cuda')
    return p.parse_args()


def main():
    args = parse_args()
    ensure_repo_on_path()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model, cfg = build_bevformer_model(args.config, args.checkpoint, device)
    patch_decoder_for_pt2_export(model)
    patch_head_bbox_for_pt2_export(model)

    bev_h = cfg.model['pts_bbox_head']['bev_h']
    bev_w = cfg.model['pts_bbox_head']['bev_w']
    embed_dims = cfg.model['pts_bbox_head']['transformer']['embed_dims']
    len_bev = bev_h * bev_w

    bev_embed = torch.randn(
        len_bev, args.batch_size, embed_dims,
        dtype=torch.float32, device=device)
    wrapper = DecoderWrapper(model, args.batch_size).eval().to(device)

    with torch.no_grad():
        ep = export(wrapper, (bev_embed,))
        ep = remove_runtime_assertions(ep)
        torch.export.save(ep, args.output)
        cls_scores, bbox_preds = wrapper(bev_embed)

    print(f'saved {args.output}')
    print(f'  all_cls_scores: {tuple(cls_scores.shape)}')
    print(f'  all_bbox_preds: {tuple(bbox_preds.shape)}')
    print(f'  (bev grid {bev_h}x{bev_w}, len_bev={len_bev})')


if __name__ == '__main__':
    main()
