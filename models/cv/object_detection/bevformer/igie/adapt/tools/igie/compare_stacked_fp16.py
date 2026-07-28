#!/usr/bin/env python3
"""Compare 3-stage fp16 IGIE stack against chained PT2 fp32 reference."""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import torch

IGIE_DIR = Path(__file__).resolve().parent
ROOT = IGIE_DIR.parents[1]
if str(IGIE_DIR) not in sys.path:
    sys.path.insert(0, str(IGIE_DIR))

from export_utils import make_full_stack_dummy_inputs  # noqa: E402
from stack_infer_core import (  # noqa: E402
    BEV_KEY,
    FEAT_KEYS,
    compare_arrays,
    load_vm,
    run_stacked_igie,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backbone-pt2', default='bevformer_backbone.pt2')
    p.add_argument('--encoder-pt2', default='bevformer_encoder.pt2')
    p.add_argument('--decoder-pt2', default='bevformer_decoder.pt2')
    p.add_argument('--bundle', default='stacked_fp16_compare_bundle.npz')
    p.add_argument('--image-shape', type=int, nargs=5, default=[1, 6, 3, 928, 1600])
    p.add_argument('--atol', type=float, default=None,
                   help='global atol (overrides per-key defaults)')
    p.add_argument('--atol-bev', type=float, default=0.01)
    p.add_argument('--atol-cls', type=float, default=0.05)
    p.add_argument('--atol-bbox', type=float, default=0.16)
    p.add_argument(
        '--backbone-so',
        default='bevformer_backbone_fp16_conv_only_ixinfer_NHWC_mdcn.so',
    )
    p.add_argument(
        '--encoder-so',
        default='bevformer_encoder_fp16_ixinfer_NHWC_msdeform.so',
    )
    p.add_argument(
        '--decoder-so',
        default='bevformer_decoder_fp16_ixinfer_NHWC_msdeform.so',
    )
    p.add_argument('--fail-on-mismatch', action='store_true')
    return p.parse_args()


def _run_stacked_pt2(
    backbone_pt2: str,
    encoder_pt2: str,
    decoder_pt2: str,
    img: torch.Tensor,
    lidar2img: torch.Tensor,
    can_bus: torch.Tensor,
    img_shape: torch.Tensor,
    device: torch.device,
) -> dict[str, np.ndarray]:
    bb = torch.export.load(backbone_pt2).module().to(device)
    enc = torch.export.load(encoder_pt2).module().to(device)
    dec = torch.export.load(decoder_pt2).module().to(device)
    with torch.no_grad():
        feats = bb(img)
        if not isinstance(feats, (tuple, list)):
            feats = (feats,)
        feat_dict = dict(zip(FEAT_KEYS, feats))
        bev_embed = enc(
            feat_dict['feat0'], feat_dict['feat1'], feat_dict['feat2'], feat_dict['feat3'],
            lidar2img, can_bus, img_shape)
        cls_scores, bbox_preds = dec(bev_embed)
    return {
        BEV_KEY: bev_embed.detach().cpu().numpy(),
        'all_cls_scores': cls_scores.detach().cpu().numpy(),
        'all_bbox_preds': bbox_preds.detach().cpu().numpy(),
    }


def _load_bundle(args, device: torch.device):
    bundle_path = ROOT / args.bundle
    if bundle_path.is_file():
        data = dict(np.load(bundle_path, allow_pickle=False))
        return (
            data['img'], data['lidar2img'], data['can_bus'], data['img_shape'],
            {
                BEV_KEY: data[BEV_KEY],
                'all_cls_scores': data['all_cls_scores'],
                'all_bbox_preds': data['all_bbox_preds'],
            },
        )

    b, n, c, h, w = args.image_shape
    dev = str(device)
    with torch.no_grad():
        img, lidar2img, can_bus, img_shape_t = make_full_stack_dummy_inputs(b, n, h, w, dev)
        ref = _run_stacked_pt2(
            str(ROOT / args.backbone_pt2),
            str(ROOT / args.encoder_pt2),
            str(ROOT / args.decoder_pt2),
            img, lidar2img, can_bus, img_shape_t, device)
    bundle = {
        'img': img.cpu().numpy(),
        'lidar2img': lidar2img.cpu().numpy(),
        'can_bus': can_bus.cpu().numpy(),
        'img_shape': img_shape_t.cpu().numpy(),
    }
    bundle.update(ref)
    np.savez(bundle_path, **bundle)
    print(f'saved bundle -> {bundle_path}')
    return bundle['img'], bundle['lidar2img'], bundle['can_bus'], bundle['img_shape'], ref


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img, lidar2img, can_bus, img_shape, ref = _load_bundle(args, device)

    print('3-stage fp16 IGIE compare (PT2 fp32 reference)')
    print(f'  backbone={args.backbone_so}')
    print(f'  encoder={args.encoder_so}')
    print(f'  decoder={args.decoder_so}')

    bb_vm, dev = load_vm(str(ROOT / args.backbone_so))
    enc_vm, _ = load_vm(str(ROOT / args.encoder_so))
    dec_vm, _ = load_vm(str(ROOT / args.decoder_so))

    out = run_stacked_igie(
        bb_vm, enc_vm, dec_vm, dev, img, lidar2img, can_bus, img_shape)

    atol_bev = args.atol if args.atol is not None else args.atol_bev
    atol_cls = args.atol if args.atol is not None else args.atol_cls
    atol_bbox = args.atol if args.atol is not None else args.atol_bbox

    ok = True
    ok &= compare_arrays(BEV_KEY, ref[BEV_KEY], out[BEV_KEY], atol_bev)
    ok &= compare_arrays(
        'all_cls_scores', ref['all_cls_scores'], out['all_cls_scores'], atol_cls)
    ok &= compare_arrays(
        'all_bbox_preds', ref['all_bbox_preds'], out['all_bbox_preds'], atol_bbox)

    del bb_vm, enc_vm, dec_vm
    gc.collect()

    print(f'\n=== result: {"PASS" if ok else "FAIL"} ===')
    if args.fail_on_mismatch and not ok:
        sys.exit(1)


if __name__ == '__main__':
    main()
