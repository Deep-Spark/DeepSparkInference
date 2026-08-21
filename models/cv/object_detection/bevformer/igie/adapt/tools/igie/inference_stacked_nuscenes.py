#!/usr/bin/env python3
"""nuScenes eval with three-stage stacked IGIE engines."""

from __future__ import annotations

import argparse
import copy
import importlib
import os
import sys
import time
from pathlib import Path

import mmcv
import numpy as np
import torch
import tvm
from mmcv import Config
from mmcv.parallel import DataContainer, collate
from mmcv.runner import load_checkpoint
from mmdet3d.core import bbox3d2result
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from torch.utils.data import DataLoader

IGIE_DIR = Path(__file__).resolve().parent
ROOT = IGIE_DIR.parents[1]
if str(IGIE_DIR) not in sys.path:
    sys.path.insert(0, str(IGIE_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from export_utils import ensure_repo_on_path  # noqa: E402
from stack_infer_core import _ndarray, load_vm, run_stacked_igie  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description='nuScenes eval with 3-stage IGIE stack')
    p.add_argument('config', help='config file path')
    p.add_argument('checkpoint', help='checkpoint for bbox decode/NMS head')
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
    p.add_argument('--data-root', default='data/nuscenes')
    p.add_argument('--ann-file', default=None)
    p.add_argument('--igie-device', default='iluvatar')
    p.add_argument('--jsonfile-prefix', default='./test/igie_stacked')
    p.add_argument('--max-samples', type=int, default=None)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--show-progress', action='store_true')
    return p.parse_args()


def _import_plugin(cfg, config_path: str):
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    if not getattr(cfg, 'plugin', False):
        return
    if hasattr(cfg, 'plugin_dir'):
        plugin_dir = cfg.plugin_dir
    else:
        plugin_dir = os.path.dirname(config_path)
    parts = Path(plugin_dir).parts
    importlib.import_module('.'.join(parts))


def _unwrap_img(img) -> torch.Tensor:
    if isinstance(img, list):
        img = img[0]
    if isinstance(img, DataContainer):
        img = img.data[0]
    while isinstance(img, (list, tuple)):
        img = img[0]
        if isinstance(img, DataContainer):
            img = img.data
    if img.dim() == 4:
        img = img.unsqueeze(0)
    return img.contiguous().float()


def _unwrap_meta_dict(meta) -> dict:
    if isinstance(meta, DataContainer):
        meta = meta.data
    return meta


def _meta_list_from_batch(data) -> list[dict]:
    img_metas = data['img_metas']
    if isinstance(img_metas, list):
        img_metas = img_metas[0]
    if isinstance(img_metas, DataContainer):
        img_metas = img_metas.data
    while isinstance(img_metas, list) and img_metas and isinstance(img_metas[0], list):
        img_metas = img_metas[0]
    return [_unwrap_meta_dict(m) for m in img_metas]


def _apply_temporal_can_bus(img_metas: list[dict], prev_frame_info: dict, video_test_mode: bool):
    metas = copy.deepcopy(img_metas)
    meta0 = metas[0]
    if meta0['scene_token'] != prev_frame_info['scene_token']:
        prev_frame_info['prev_bev'] = None
    prev_frame_info['scene_token'] = meta0['scene_token']

    if not video_test_mode:
        prev_frame_info['prev_bev'] = None

    tmp_pos = copy.deepcopy(meta0['can_bus'][:3])
    tmp_angle = copy.deepcopy(meta0['can_bus'][-1])
    if prev_frame_info['prev_bev'] is not None:
        meta0['can_bus'][:3] -= prev_frame_info['prev_pos']
        meta0['can_bus'][-1] -= prev_frame_info['prev_angle']
    else:
        meta0['can_bus'][-1] = 0
        meta0['can_bus'][:3] = 0

    prev_frame_info['prev_pos'] = tmp_pos
    prev_frame_info['prev_angle'] = tmp_angle
    return metas


def _build_igie_meta_tensors(img_metas: list[dict], device: torch.device):
    batch_size = len(img_metas)
    num_cams = len(img_metas[0]['lidar2img'])
    lidar2img = torch.stack([
        torch.stack([torch.from_numpy(m['lidar2img'][c]).float() for c in range(num_cams)], dim=0)
        for m in img_metas
    ], dim=0).to(device)
    can_bus = torch.stack([
        torch.from_numpy(np.asarray(m['can_bus'], dtype=np.float32)) for m in img_metas
    ], dim=0).to(device)
    img_shapes = []
    for m in img_metas:
        cam_shapes = []
        for cam_shape in m['img_shape']:
            cam_shapes.append([float(cam_shape[0]), float(cam_shape[1])])
        img_shapes.append(torch.tensor(cam_shapes, dtype=torch.float32))
    img_shape = torch.stack(img_shapes, dim=0).to(device)
    return lidar2img, can_bus, img_shape


def main():
    args = parse_args()
    ensure_repo_on_path()

    cfg = Config.fromfile(args.config)
    _import_plugin(cfg, args.config)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None

    val_cfg = cfg.data.val.copy()
    val_cfg['test_mode'] = True
    val_cfg.pop('samples_per_gpu', None)
    if args.ann_file is not None:
        val_cfg['ann_file'] = args.ann_file
    else:
        val_cfg['ann_file'] = os.path.join(args.data_root, 'nuscenes_infos_temporal_val.pkl')
    val_cfg['data_root'] = args.data_root

    dataset = build_dataset(val_cfg)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=lambda batch: collate(batch, samples_per_gpu=1),
    )

    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.eval()
    head = model.pts_bbox_head
    video_test_mode = bool(cfg.model.get('video_test_mode', False))

    for name, so in (
        ('backbone', args.backbone_so),
        ('encoder', args.encoder_so),
        ('decoder', args.decoder_so),
    ):
        path = ROOT / so
        if not path.is_file():
            raise SystemExit(f'missing {name} engine: {path}')

    bb_vm, dev = load_vm(str(ROOT / args.backbone_so), args.igie_device)
    enc_vm, _ = load_vm(str(ROOT / args.encoder_so), args.igie_device)
    dec_vm, _ = load_vm(str(ROOT / args.decoder_so), args.igie_device)

    print(f'stacked engines:')
    print(f'  backbone={args.backbone_so}')
    print(f'  encoder={args.encoder_so}')
    print(f'  decoder={args.decoder_so}')

    prev_frame_info = {
        'prev_bev': None,
        'scene_token': None,
        'prev_pos': 0,
        'prev_angle': 0,
    }

    bbox_results = []
    total = len(dataset) if args.max_samples is None else min(args.max_samples, len(dataset))
    t0 = time.time()

    for step, data in enumerate(loader):
        if args.max_samples is not None and step >= args.max_samples:
            break

        img = _unwrap_img(data['img']).cuda()
        img_metas = _apply_temporal_can_bus(
            _meta_list_from_batch(data), prev_frame_info, video_test_mode)

        lidar2img, can_bus, img_shape = _build_igie_meta_tensors(img_metas, img.device)
        stacked = run_stacked_igie(
            bb_vm, enc_vm, dec_vm, dev,
            img.cpu().numpy(),
            lidar2img.cpu().numpy(),
            can_bus.cpu().numpy(),
            img_shape.cpu().numpy(),
        )
        preds = {
            'all_cls_scores': torch.from_numpy(stacked['all_cls_scores']).to(device=img.device),
            'all_bbox_preds': torch.from_numpy(stacked['all_bbox_preds']).to(device=img.device),
            'bev_embed': torch.from_numpy(stacked['bev_embed']).to(device=img.device),
        }

        bbox_list = head.get_bboxes(preds, img_metas, rescale=True)
        for bboxes, scores, labels in bbox_list:
            bbox_results.append({'pts_bbox': bbox3d2result(bboxes, scores, labels)})

        if args.show_progress and (step + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f'[{step + 1}/{total}] {elapsed:.1f}s', flush=True)

    elapsed = time.time() - t0
    print(f'inference done: {len(bbox_results)} samples in {elapsed:.1f}s '
          f'({len(bbox_results) / max(elapsed, 1e-6):.2f} fps)')

    if args.max_samples is not None and len(bbox_results) < len(dataset):
        print(f'skip nuScenes eval: only {len(bbox_results)}/{len(dataset)} samples')
        return

    eval_kwargs = cfg.get('evaluation', {}).copy()
    for key in ('interval', 'tmpdir', 'start', 'gpu_collect', 'save_best', 'rule'):
        eval_kwargs.pop(key, None)
    eval_kwargs.update(metric='bbox', jsonfile_prefix=args.jsonfile_prefix)

    metrics = dataset.evaluate(bbox_results, **eval_kwargs)
    print('\n=== nuScenes metrics (IGIE 3-stage stack) ===')
    for k, v in sorted(metrics.items()):
        if 'mAP' in k or 'NDS' in k:
            print(f'  {k}: {v}')


if __name__ == '__main__':
    main()
