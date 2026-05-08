# Copyright (c) 2024, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.
#
# This file replaces the mmdet/mmengine Runner approach with a standalone
# implementation using pycocotools + cv2 + torchvision.ops.

import math
import os
import time
import argparse

import cv2
import numpy as np
import torch
import torchvision.ops as tv_ops
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import tensorrt
from cuda import cudart

from common import create_engine_context, get_io_bindings


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=str, required=True,
                        help="TensorRT engine path.")
    parser.add_argument("--cfg_file", type=str, default="",
                        help="MMDet Python config file path.")
    parser.add_argument("--batchsize", type=int, required=True,
                        help="Inference batch size.")
    parser.add_argument("--datasets", type=str, required=True,
                        help="COCO dataset root (contains val2017/ and annotations/).")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Number of warmup iterations.")
    parser.add_argument("--acc_target", type=float, default=None)
    parser.add_argument("--fps_target", type=float, default=None)
    parser.add_argument("--perf_only", type=bool, default=False,
                        help="Run performance test only.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_cfg(cfg_file):
    """Execute a Python config file and return its namespace dict."""
    ns = {}
    with open(cfg_file) as f:
        exec(compile(f.read(), cfg_file, 'exec'), ns)  # noqa: S102
    return ns


def extract_cfg_params(cfg_ns):
    """Extract preprocessing and model parameters from config namespace."""
    model = cfg_ns.get('model', {})

    # Data preprocessor
    pp = model.get('data_preprocessor', {})
    mean = pp.get('mean', [123.675, 116.28, 103.53])
    std = pp.get('std', [58.395, 57.12, 57.375])
    bgr_to_rgb = pp.get('bgr_to_rgb', True)
    pad_divisor = pp.get('pad_size_divisor', 32)

    # Head type and strides
    head_cfg = model.get('bbox_head', {})
    head_type = head_cfg.get('type', 'FCOSHead')
    strides = head_cfg.get('strides', [8, 16, 32, 64, 128])
    norm_on_bbox = head_cfg.get('norm_on_bbox', False)
    num_classes = head_cfg.get('num_classes', 80)

    # FoveaBox-specific
    base_edge_list = head_cfg.get('base_edge_list', [16, 32, 64, 128, 256])

    # FSAF-specific: TBLRBBoxCoder normalizer
    bbox_coder = head_cfg.get('bbox_coder', {})
    tblr_normalizer = bbox_coder.get('normalizer', 4.0)

    # Test cfg
    test_cfg = model.get('test_cfg', {})

    # Resize parameters: prefer val_dataloader (single-image accuracy evaluation),
    # fall back to test_dataloader (batch performance testing).
    vd = cfg_ns.get('val_dataloader', {})
    val_pipeline = vd.get('dataset', {}).get('pipeline', [])
    td = cfg_ns.get('test_dataloader', {})
    test_pipeline = td.get('dataset', {}).get('pipeline', [])
    pipeline = val_pipeline if val_pipeline else test_pipeline
    scale = (800, 800)
    keep_ratio = True
    for p in pipeline:
        if isinstance(p, dict) and p.get('type') == 'Resize':
            scale = p.get('scale', (800, 800))
            keep_ratio = p.get('keep_ratio', True)
            break

    return {
        'mean': mean, 'std': std, 'bgr_to_rgb': bgr_to_rgb,
        'pad_divisor': pad_divisor,
        'head_type': head_type, 'strides': strides,
        'norm_on_bbox': norm_on_bbox, 'num_classes': num_classes,
        'base_edge_list': base_edge_list, 'tblr_normalizer': tblr_normalizer,
        'test_cfg': test_cfg,
        'scale': scale, 'keep_ratio': keep_ratio,
    }


# ---------------------------------------------------------------------------
# COCO dataset
# ---------------------------------------------------------------------------

class CocoDetDataset(Dataset):
    """Loads COCO val images with preprocessing."""

    def __init__(self, coco_root, mean, std, bgr_to_rgb, pad_divisor,
                 scale, keep_ratio):
        ann_file = os.path.join(coco_root,
                                'annotations/instances_val2017.json')
        self.coco = COCO(ann_file)
        self.img_ids = sorted(self.coco.getImgIds())
        self.img_dir = os.path.join(coco_root, 'val2017')
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.bgr_to_rgb = bgr_to_rgb
        self.pad_divisor = pad_divisor
        self.scale = scale         # (h, w) maximum size
        self.keep_ratio = keep_ratio

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])

        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f'Image not found: {img_path}')

        orig_h, orig_w = img_bgr.shape[:2]
        target_h, target_w = int(self.scale[0]), int(self.scale[1])

        if self.keep_ratio:
            sf = min(target_h / orig_h, target_w / orig_w)
            new_h = int(orig_h * sf + 0.5)
            new_w = int(orig_w * sf + 0.5)
            scale_factor = float(sf)
        else:
            new_h, new_w = target_h, target_w
            sf_w = new_w / orig_w
            sf_h = new_h / orig_h
            scale_factor = np.array([sf_w, sf_h, sf_w, sf_h], dtype=np.float32)

        img = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        img_shape = (new_h, new_w)

        if self.bgr_to_rgb:
            img = img[:, :, ::-1]

        img = img.astype(np.float32)
        img = (img - self.mean) / self.std

        # Pad to divisor, and ensure at least target size (engine may have fixed spatial dims)
        pad_h = max(math.ceil(new_h / self.pad_divisor) * self.pad_divisor, target_h)
        pad_w = max(math.ceil(new_w / self.pad_divisor) * self.pad_divisor, target_w)
        if pad_h != new_h or pad_w != new_w:
            padded = np.zeros((pad_h, pad_w, 3), dtype=np.float32)
            padded[:new_h, :new_w] = img
            img = padded

        img_tensor = torch.from_numpy(img.transpose(2, 0, 1))  # CHW

        meta = {
            'img_id': img_id,
            'ori_shape': (orig_h, orig_w),
            'img_shape': img_shape,
            'scale_factor': scale_factor,
        }
        return img_tensor, meta


def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    metas = [b[1] for b in batch]
    return imgs, metas


# ---------------------------------------------------------------------------
# FPN prior point generation
# ---------------------------------------------------------------------------

def _make_points(H, W, stride, center_offset=0.5):
    """Generate center points for a feature map of size (H, W) at given stride.

    center_offset=0.5: anchor-free models (FCOS, FoveaBox) place points at the
                       center of each cell — (col + 0.5) * stride.
    center_offset=0.0: anchor-based models (FSAF) follow mmdet3 AnchorGenerator
                       default, placing anchor centers at col * stride.
    """
    ys = (torch.arange(H, dtype=torch.float32) + center_offset) * stride
    xs = (torch.arange(W, dtype=torch.float32) + center_offset) * stride
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)  # [H*W, 2]


# ---------------------------------------------------------------------------
# Per-image decode and NMS helpers
# ---------------------------------------------------------------------------

def _apply_nms_and_limit(bboxes, scores, labels, nms_iou_thr, max_per_img):
    if len(bboxes) == 0:
        return bboxes, scores, labels
    keep = tv_ops.nms(bboxes.float(), scores.float(), nms_iou_thr)
    bboxes, scores, labels = bboxes[keep], scores[keep], labels[keep]
    if max_per_img > 0 and len(bboxes) > max_per_img:
        _, topk = scores.topk(max_per_img)
        bboxes, scores, labels = bboxes[topk], scores[topk], labels[topk]
    return bboxes, scores, labels


def _rescale_bboxes(bboxes, scale_factor, ori_shape):
    if isinstance(scale_factor, np.ndarray):
        sf = torch.from_numpy(scale_factor).float()
        bboxes = bboxes / sf
    else:
        bboxes = bboxes / float(scale_factor)
    orig_h, orig_w = ori_shape
    bboxes[:, 0::2] = bboxes[:, 0::2].clamp(0, orig_w)
    bboxes[:, 1::2] = bboxes[:, 1::2].clamp(0, orig_h)
    return bboxes


def _top_nms_pre(nms_pre, scores_1d, *tensors):
    """Select top-nms_pre candidates per level."""
    if nms_pre > 0 and scores_1d.shape[0] > nms_pre:
        _, idx = scores_1d.topk(nms_pre)
        return (t[idx] for t in tensors)
    return iter(tensors)


# ---------------------------------------------------------------------------
# FCOS / HRNet (FCOSHead) decode
# ---------------------------------------------------------------------------

def decode_fcos_single(cls_scores, bbox_preds, score_factors,
                       strides, img_shape, ori_shape, scale_factor,
                       test_cfg, norm_on_bbox=False):
    """Decode one image from FCOS head outputs."""
    nms_pre = test_cfg.get('nms_pre', 1000)
    score_thr = test_cfg.get('score_thr', 0.05)
    nms_iou_thr = test_cfg.get('nms', {}).get('iou_threshold', 0.5)
    max_per_img = test_cfg.get('max_per_img', 100)
    min_bbox_size = test_cfg.get('min_bbox_size', 0)
    img_h, img_w = img_shape

    all_bboxes, all_scores, all_labels = [], [], []

    for i, stride in enumerate(strides):
        cls = cls_scores[i].sigmoid()               # [C, H, W]
        bbox = bbox_preds[i]                        # [4, H, W]
        ctr = score_factors[i].sigmoid() if score_factors is not None else None

        C, H, W = cls.shape
        cls = cls.permute(1, 2, 0).reshape(-1, C)   # [HW, C]
        bbox = bbox.permute(1, 2, 0).reshape(-1, 4) # [HW, 4]
        if ctr is not None:
            ctr = ctr.permute(1, 2, 0).reshape(-1)  # [HW]

        points = _make_points(H, W, stride)          # [HW, 2]

        # Score = cls × centerness
        if ctr is not None:
            max_scores = (cls * ctr.unsqueeze(-1)).max(dim=-1).values
            labels = cls.max(dim=-1).indices
        else:
            max_scores, labels = cls.max(dim=-1)

        # Top-k selection
        if nms_pre > 0 and max_scores.shape[0] > nms_pre:
            _, topk_idx = max_scores.topk(nms_pre)
            points = points[topk_idx]
            bbox = bbox[topk_idx]
            if ctr is not None:
                ctr = ctr[topk_idx]
            labels = labels[topk_idx]
            max_scores = max_scores[topk_idx]

        # Decode distances → xyxy
        dist = bbox * stride if norm_on_bbox else bbox
        px, py = points[:, 0], points[:, 1]
        x1 = (px - dist[:, 0]).clamp(0, img_w)
        y1 = (py - dist[:, 1]).clamp(0, img_h)
        x2 = (px + dist[:, 2]).clamp(0, img_w)
        y2 = (py + dist[:, 3]).clamp(0, img_h)
        bboxes = torch.stack([x1, y1, x2, y2], dim=-1)

        # Filter by score and min size
        keep = max_scores > score_thr
        if min_bbox_size > 0:
            bw = (bboxes[:, 2] - bboxes[:, 0])
            bh = (bboxes[:, 3] - bboxes[:, 1])
            keep &= (bw >= min_bbox_size) & (bh >= min_bbox_size)
        bboxes = bboxes[keep]
        labels = labels[keep]
        max_scores = max_scores[keep]

        all_bboxes.append(bboxes)
        all_scores.append(max_scores)
        all_labels.append(labels)

    if not all_bboxes:
        return (torch.zeros((0, 4)), torch.zeros(0),
                torch.zeros(0, dtype=torch.long))

    bboxes = torch.cat(all_bboxes)
    scores = torch.cat(all_scores)
    labels = torch.cat(all_labels)

    bboxes, scores, labels = _apply_nms_and_limit(
        bboxes, scores, labels, nms_iou_thr, max_per_img)
    bboxes = _rescale_bboxes(bboxes, scale_factor, ori_shape)
    return bboxes, scores, labels


# ---------------------------------------------------------------------------
# FoveaBox decode
# ---------------------------------------------------------------------------

def decode_fovea_single(cls_scores, bbox_preds, strides, base_edge_list,
                        img_shape, ori_shape, scale_factor, test_cfg):
    """Decode one image from FoveaHead outputs."""
    nms_pre = test_cfg.get('nms_pre', 1000)
    score_thr = test_cfg.get('score_thr', 0.05)
    nms_iou_thr = test_cfg.get('nms', {}).get('iou_threshold', 0.5)
    max_per_img = test_cfg.get('max_per_img', 100)
    img_h, img_w = img_shape

    all_bboxes, all_scores, all_labels = [], [], []

    for i, stride in enumerate(strides):
        cls = cls_scores[i].sigmoid()               # [C, H, W]
        bbox = bbox_preds[i]                        # [4, H, W]
        base_len = base_edge_list[i]

        C, H, W = cls.shape
        cls = cls.permute(1, 2, 0).reshape(-1, C)
        bbox = bbox.permute(1, 2, 0).reshape(-1, 4)

        points = _make_points(H, W, stride)
        max_scores, labels = cls.max(dim=-1)

        if nms_pre > 0 and max_scores.shape[0] > nms_pre:
            _, topk_idx = max_scores.topk(nms_pre)
            points = points[topk_idx]
            bbox = bbox[topk_idx]
            labels = labels[topk_idx]
            max_scores = max_scores[topk_idx]

        # FoveaBox decode: the ONNX model outputs are in log-scale
        # (exp() is NOT applied inside the ONNX graph). Apply exp() here.
        # The regression target normalization is (offset / base_edge_list[i]),
        # so the decode factor is base_edge_list[i] (not base_edge/2).
        factor = base_len
        px, py = points[:, 0], points[:, 1]
        x1 = (px - factor * bbox[:, 0].exp()).clamp(0, img_w)
        y1 = (py - factor * bbox[:, 1].exp()).clamp(0, img_h)
        x2 = (px + factor * bbox[:, 2].exp()).clamp(0, img_w)
        y2 = (py + factor * bbox[:, 3].exp()).clamp(0, img_h)
        bboxes = torch.stack([x1, y1, x2, y2], dim=-1)

        keep = max_scores > score_thr
        bboxes = bboxes[keep]
        labels = labels[keep]
        max_scores = max_scores[keep]

        all_bboxes.append(bboxes)
        all_scores.append(max_scores)
        all_labels.append(labels)

    if not all_bboxes:
        return (torch.zeros((0, 4)), torch.zeros(0),
                torch.zeros(0, dtype=torch.long))

    bboxes = torch.cat(all_bboxes)
    scores = torch.cat(all_scores)
    labels = torch.cat(all_labels)

    bboxes, scores, labels = _apply_nms_and_limit(
        bboxes, scores, labels, nms_iou_thr, max_per_img)
    bboxes = _rescale_bboxes(bboxes, scale_factor, ori_shape)
    return bboxes, scores, labels


# ---------------------------------------------------------------------------
# FSAF (TBLRBBoxCoder) decode
# ---------------------------------------------------------------------------

def decode_fsaf_single(cls_scores, bbox_preds, strides, normalizer,
                       img_shape, ori_shape, scale_factor, test_cfg):
    """Decode one image from FSAFHead (TBLRBBoxCoder) outputs."""
    nms_pre = test_cfg.get('nms_pre', 1000)
    score_thr = test_cfg.get('score_thr', 0.05)
    nms_iou_thr = test_cfg.get('nms', {}).get('iou_threshold', 0.5)
    max_per_img = test_cfg.get('max_per_img', 100)
    min_bbox_size = test_cfg.get('min_bbox_size', 0)
    img_h, img_w = img_shape

    all_bboxes, all_scores, all_labels = [], [], []

    for i, stride in enumerate(strides):
        cls = cls_scores[i].sigmoid()               # [C, H, W]
        bbox = bbox_preds[i]                        # [4, H, W] (t, b, l, r)

        C, H, W = cls.shape
        cls = cls.permute(1, 2, 0).reshape(-1, C)
        bbox = bbox.permute(1, 2, 0).reshape(-1, 4)

        # mmdet3 AnchorGenerator default center_offset=0.0: anchor centers at
        # (col * stride, row * stride), NOT (col+0.5)*stride.
        points = _make_points(H, W, stride, center_offset=0.0)
        max_scores, labels = cls.max(dim=-1)

        if nms_pre > 0 and max_scores.shape[0] > nms_pre:
            _, topk_idx = max_scores.topk(nms_pre)
            points = points[topk_idx]
            bbox = bbox[topk_idx]
            labels = labels[topk_idx]
            max_scores = max_scores[topk_idx]

        # TBLR decode: anchor is stride-sized square at each grid point
        # TBLRBBoxCoder: pred_l = (cx - x1) / (normalizer * anchor_w)
        # → x1 = cx - normalizer * anchor_w * pred_l  (anchor_w = stride)
        # pred order: (top, bottom, left, right)
        factor = normalizer * stride
        px, py = points[:, 0], points[:, 1]
        x1 = (px - bbox[:, 2] * factor).clamp(0, img_w)   # left
        y1 = (py - bbox[:, 0] * factor).clamp(0, img_h)   # top
        x2 = (px + bbox[:, 3] * factor).clamp(0, img_w)   # right
        y2 = (py + bbox[:, 1] * factor).clamp(0, img_h)   # bottom
        bboxes = torch.stack([x1, y1, x2, y2], dim=-1)

        keep = max_scores > score_thr
        if min_bbox_size > 0:
            bw = bboxes[:, 2] - bboxes[:, 0]
            bh = bboxes[:, 3] - bboxes[:, 1]
            keep &= (bw >= min_bbox_size) & (bh >= min_bbox_size)
        bboxes = bboxes[keep]
        labels = labels[keep]
        max_scores = max_scores[keep]

        all_bboxes.append(bboxes)
        all_scores.append(max_scores)
        all_labels.append(labels)

    if not all_bboxes:
        return (torch.zeros((0, 4)), torch.zeros(0),
                torch.zeros(0, dtype=torch.long))

    bboxes = torch.cat(all_bboxes)
    scores = torch.cat(all_scores)
    labels = torch.cat(all_labels)

    bboxes, scores, labels = _apply_nms_and_limit(
        bboxes, scores, labels, nms_iou_thr, max_per_img)
    bboxes = _rescale_bboxes(bboxes, scale_factor, ori_shape)
    return bboxes, scores, labels


# ---------------------------------------------------------------------------
# Output routing
# ---------------------------------------------------------------------------

def _is_fovea_or_fsaf(cfg_file):
    name = os.path.basename(cfg_file).lower()
    return name.startswith(('fovea_', 'fsaf_'))


def gather_outputs(raw_outputs, cfg_file, num_classes):
    """Sort TRT output tensors into cls_score, bbox_pred, score_factors lists."""
    cls_score, box_reg, score_factors = [], [], []
    if _is_fovea_or_fsaf(cfg_file):
        half = len(raw_outputs) // 2
        for i, out in enumerate(raw_outputs):
            if i < half:
                cls_score.append(out)
            else:
                box_reg.append(out)
    else:
        for out in raw_outputs:
            if out.shape[1] == num_classes:
                cls_score.append(out)
            elif out.shape[1] == 4:
                box_reg.append(out)
            else:
                score_factors.append(out)
    return cls_score, box_reg, score_factors


# ---------------------------------------------------------------------------
# COCO label mapping: class index (0-based) → COCO category ID
# ---------------------------------------------------------------------------

def build_label_map(coco):
    """Map 0-based class indices to COCO category IDs (sorted by ID)."""
    cat_ids = sorted(coco.getCatIds())
    return {i: cat_id for i, cat_id in enumerate(cat_ids)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    batch_size = args.batchsize

    logger = tensorrt.Logger(tensorrt.Logger.ERROR)
    engine, context = create_engine_context(args.engine, logger)
    inputs, outputs, allocations = get_io_bindings(engine)

    # Warmup
    if args.warmup > 0:
        print("\nWarm Start.")
        for _ in range(args.warmup):
            context.execute_v2(allocations)
        print("Warm Done.")

    # Performance-only mode
    if args.perf_only:
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(10):
            context.execute_v2(allocations)
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        fps = 10 * batch_size / elapsed
        print("FPS :", fps)
        print(f"Performance Check : Test {fps} >= target {args.fps_target}")
        if fps >= args.fps_target:
            print("pass!")
            exit()
        else:
            print("failed!")
            exit(1)

    # Load config
    cfg_ns = load_cfg(args.cfg_file)
    params = extract_cfg_params(cfg_ns)

    # Override dataset path
    coco_root = args.datasets

    # Build dataset and dataloader
    dataset = CocoDetDataset(
        coco_root,
        mean=params['mean'],
        std=params['std'],
        bgr_to_rgb=params['bgr_to_rgb'],
        pad_divisor=params['pad_divisor'],
        scale=params['scale'],
        keep_ratio=params['keep_ratio'],
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        drop_last=False,
    )

    coco = dataset.coco
    label_map = build_label_map(coco)

    head_type = params['head_type']
    strides = params['strides']
    test_cfg = params['test_cfg']
    norm_on_bbox = params['norm_on_bbox']
    num_classes = params['num_classes']
    base_edge_list = params['base_edge_list']
    tblr_normalizer = params['tblr_normalizer']

    is_fovea_fsaf = _is_fovea_or_fsaf(args.cfg_file)

    coco_results = []

    for imgs, metas in tqdm(loader):
        real_bs = imgs.shape[0]
        pad_batch = real_bs != batch_size

        img_np = imgs.numpy().astype(inputs[0]["dtype"])
        if pad_batch:
            padded = np.zeros((batch_size, *img_np.shape[1:]),
                              dtype=inputs[0]["dtype"])
            padded[:real_bs] = img_np
            img_np = padded
        img_np = np.ascontiguousarray(img_np)

        # Copy to GPU and run inference
        (err,) = cudart.cudaMemcpy(
            inputs[0]["allocation"], img_np, img_np.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        assert err == cudart.cudaError_t.cudaSuccess

        context.execute_v2(allocations)

        # Retrieve outputs
        raw_outputs = []
        for out_info in outputs:
            out = np.zeros(out_info["shape"], out_info["dtype"])
            (err,) = cudart.cudaMemcpy(
                out, out_info["allocation"], out_info["nbytes"],
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
            assert err == cudart.cudaError_t.cudaSuccess
            out = torch.from_numpy(out[:real_bs] if pad_batch else out)
            raw_outputs.append(out)

        cls_score, box_reg, score_factors = gather_outputs(
            raw_outputs, args.cfg_file, num_classes)

        # Decode per image
        for img_idx, meta in enumerate(metas):
            img_id = meta['img_id']
            img_shape = meta['img_shape']       # (new_h, new_w)
            ori_shape = meta['ori_shape']       # (orig_h, orig_w)
            scale_factor = meta['scale_factor']

            # Extract per-image outputs
            cls_i = [c[img_idx] for c in cls_score]    # list of [C, H, W]
            bbox_i = [b[img_idx] for b in box_reg]     # list of [4, H, W]
            sf_i = [s[img_idx] for s in score_factors] if score_factors else None

            if head_type == 'FCOSHead':
                bboxes, scores, labels = decode_fcos_single(
                    cls_i, bbox_i, sf_i, strides, img_shape, ori_shape,
                    scale_factor, test_cfg, norm_on_bbox)
            elif head_type == 'FoveaHead':
                bboxes, scores, labels = decode_fovea_single(
                    cls_i, bbox_i, strides, base_edge_list,
                    img_shape, ori_shape, scale_factor, test_cfg)
            elif head_type == 'FSAFHead':
                bboxes, scores, labels = decode_fsaf_single(
                    cls_i, bbox_i, strides, tblr_normalizer,
                    img_shape, ori_shape, scale_factor, test_cfg)
            else:
                # Fallback: treat as FCOSHead
                bboxes, scores, labels = decode_fcos_single(
                    cls_i, bbox_i, sf_i, strides, img_shape, ori_shape,
                    scale_factor, test_cfg, norm_on_bbox)

            # Convert to COCO result format
            bboxes_np = bboxes.numpy()
            scores_np = scores.numpy()
            labels_np = labels.numpy()

            for det_idx in range(len(bboxes_np)):
                x1, y1, x2, y2 = bboxes_np[det_idx]
                score = float(scores_np[det_idx])
                label = int(labels_np[det_idx])
                cat_id = label_map.get(label, label + 1)

                coco_results.append({
                    'image_id': int(img_id),
                    'category_id': cat_id,
                    'bbox': [float(x1), float(y1),
                             float(x2 - x1), float(y2 - y1)],
                    'score': score,
                })

    # Evaluate with pycocotools
    if not coco_results:
        print("No detections produced.")
        return

    coco_dt = coco.loadRes(coco_results)
    evaluator = COCOeval(coco, coco_dt, iouType='bbox')
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    ap50_95 = evaluator.stats[0]
    ap50 = evaluator.stats[1]

    print(f"\n[Accuracy] mAP@0.5:0.95 = {ap50_95:.4f}, mAP@0.5 = {ap50:.4f}")

    if args.acc_target is not None:
        if ap50_95 >= args.acc_target:
            print("pass!")
        else:
            print("failed!")
            exit(1)


if __name__ == "__main__":
    main()