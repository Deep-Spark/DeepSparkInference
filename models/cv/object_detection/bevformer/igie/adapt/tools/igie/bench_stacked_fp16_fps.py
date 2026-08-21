#!/usr/bin/env python3
"""Bench BEVFormer 3-stage fp16 stack: Torch vs IGIE wall time."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

IGIE_DIR = Path(__file__).resolve().parent
ROOT = IGIE_DIR.parents[1]
if str(IGIE_DIR) not in sys.path:
    sys.path.insert(0, str(IGIE_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from export_utils import (  # noqa: E402
    build_bevformer_model,
    ensure_repo_on_path,
    extract_image_features,
)
from stack_infer_core import (  # noqa: E402
    load_vm,
    run_backbone_igie,
    run_decoder_igie,
    run_encoder_igie,
    run_stacked_igie,
)


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _ms(fn, warmup: int, runs: int) -> float:
    for _ in range(warmup):
        fn()
    _sync()
    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    _sync()
    return (time.perf_counter() - t0) * 1000.0 / max(runs, 1)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='projects/configs/bevformer/bevformer-base.py')
    p.add_argument('--checkpoint', default='ckpts/bevformer-base.pth')
    p.add_argument('--bundle', default='stacked_fp16_compare_bundle.npz')
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
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--runs', type=int, default=10)
    p.add_argument('--dev-id', type=int, default=0)
    p.add_argument('--skip-torch', action='store_true')
    p.add_argument('--skip-igie', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    ensure_repo_on_path()

    bundle = ROOT / args.bundle
    if not bundle.is_file():
        raise SystemExit(f'missing bundle: {bundle}')
    data = dict(np.load(bundle, allow_pickle=False))
    img = data['img']
    lidar2img = data['lidar2img']
    can_bus = data['can_bus']
    img_shape = data['img_shape']

    results = {}

    if not args.skip_torch:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model, _ = build_bevformer_model(args.config, args.checkpoint, str(device))
        model.use_grid_mask = False
        model.eval()
        img_t = torch.from_numpy(img).to(device)
        batch_size, num_cams = img.shape[0], img.shape[1]

        # Native torch head expects numpy/cpu scalars in img_metas.
        img_metas = []
        for b in range(batch_size):
            img_metas.append(
                {
                    'lidar2img': np.ascontiguousarray(lidar2img[b]),
                    'can_bus': np.ascontiguousarray(can_bus[b]),
                    'img_shape': np.ascontiguousarray(img_shape[b]),
                }
            )

        def _torch_backbone():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                feats = extract_image_features(model, img_t)
                for feat in feats:
                    if torch.is_tensor(feat) and feat.is_cuda:
                        _ = feat.reshape(-1)[0].item()
            return feats

        def _torch_head(feats):
            with torch.cuda.amp.autocast(dtype=torch.float16):
                mlvl = []
                for feat in feats:
                    mlvl.append(
                        feat.reshape(
                            batch_size,
                            num_cams,
                            feat.shape[1],
                            feat.shape[2],
                            feat.shape[3],
                        )
                    )
                outputs = model.pts_bbox_head(mlvl, img_metas, prev_bev=None)
                for key in ('all_cls_scores', 'all_bbox_preds', 'bev_embed'):
                    x = outputs[key]
                    if torch.is_tensor(x) and x.is_cuda:
                        _ = x.reshape(-1)[0].item()
            return outputs

        def _torch_run():
            feats = _torch_backbone()
            return _torch_head(feats)

        torch_bb_ms = _ms(_torch_backbone, args.warmup, args.runs)
        feats0 = _torch_backbone()
        torch_head_ms = _ms(lambda: _torch_head(feats0), args.warmup, args.runs)
        torch_ms = _ms(_torch_run, args.warmup, args.runs)
        results['torch_backbone_ms'] = torch_bb_ms
        results['torch_head_ms'] = torch_head_ms
        results['torch_e2e_ms'] = torch_ms
        print(f'torch_backbone_ms={torch_bb_ms:.3f}')
        print(f'torch_head_ms={torch_head_ms:.3f}')
        print(f'torch_e2e_ms={torch_ms:.3f}')
        print(f'torch_stages_sum_ms={torch_bb_ms + torch_head_ms:.3f}')
        del model
        torch.cuda.empty_cache()

    if not args.skip_igie:
        for name, path in (
            ('backbone', ROOT / args.backbone_so),
            ('encoder', ROOT / args.encoder_so),
            ('decoder', ROOT / args.decoder_so),
        ):
            if not path.is_file():
                raise SystemExit(f'missing {name} engine: {path}')

        bb_vm, dev = load_vm(str(ROOT / args.backbone_so), 'iluvatar', args.dev_id)
        enc_vm, _ = load_vm(str(ROOT / args.encoder_so), 'iluvatar', args.dev_id)
        dec_vm, _ = load_vm(str(ROOT / args.decoder_so), 'iluvatar', args.dev_id)

        def _bb():
            return run_backbone_igie(bb_vm, dev, img)

        def _enc(feats):
            return run_encoder_igie(enc_vm, dev, feats, lidar2img, can_bus, img_shape)

        def _dec(bev):
            return run_decoder_igie(dec_vm, dev, bev)

        def _e2e():
            return run_stacked_igie(
                bb_vm, enc_vm, dec_vm, dev, img, lidar2img, can_bus, img_shape)

        bb_ms = _ms(_bb, args.warmup, args.runs)
        feats = _bb()
        enc_ms = _ms(lambda: _enc(feats), args.warmup, args.runs)
        bev = _enc(feats)
        dec_ms = _ms(lambda: _dec(bev), args.warmup, args.runs)
        e2e_ms = _ms(_e2e, args.warmup, args.runs)

        results.update(
            {
                'igie_backbone_ms': bb_ms,
                'igie_encoder_ms': enc_ms,
                'igie_decoder_ms': dec_ms,
                'igie_e2e_ms': e2e_ms,
                'igie_stages_sum_ms': bb_ms + enc_ms + dec_ms,
            }
        )
        print(f'igie_backbone_ms={bb_ms:.3f}')
        print(f'igie_encoder_ms={enc_ms:.3f}')
        print(f'igie_decoder_ms={dec_ms:.3f}')
        print(f'igie_stages_sum_ms={bb_ms + enc_ms + dec_ms:.3f}')
        print(f'igie_e2e_ms={e2e_ms:.3f}')
        print(f'igie engines: backbone={args.backbone_so}')
        print(f'              encoder={args.encoder_so}')
        print(f'              decoder={args.decoder_so}')

    if 'torch_e2e_ms' in results and 'igie_e2e_ms' in results:
        ratio = results['igie_e2e_ms'] / max(results['torch_e2e_ms'], 1e-6)
        print(
            f'ratio_igie/torch={ratio:.3f}x '
            f'(torch={results["torch_e2e_ms"]:.1f}ms, igie={results["igie_e2e_ms"]:.1f}ms)'
        )
        if 'torch_backbone_ms' in results and 'igie_backbone_ms' in results:
            print(
                f'ratio_backbone={results["igie_backbone_ms"] / max(results["torch_backbone_ms"], 1e-6):.3f}x '
                f'(torch={results["torch_backbone_ms"]:.1f}ms, igie={results["igie_backbone_ms"]:.1f}ms)'
            )
            # Torch head ≈ IGIE encoder+decoder (not 1:1 split).
            torch_tail = results['torch_head_ms']
            igie_tail = results['igie_encoder_ms'] + results['igie_decoder_ms']
            print(
                f'ratio_head_vs_enc+dec={igie_tail / max(torch_tail, 1e-6):.3f}x '
                f'(torch_head={torch_tail:.1f}ms, igie_enc+dec={igie_tail:.1f}ms)'
            )


if __name__ == '__main__':
    main()
