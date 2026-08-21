"""Three-stage stacked IGIE inference: backbone → encoder → decoder."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import tvm
from tvm import relax

FEAT_KEYS = ('feat0', 'feat1', 'feat2', 'feat3')
META_KEYS = ('lidar2img', 'can_bus', 'img_shape')
ENCODER_INPUT_KEYS = FEAT_KEYS + META_KEYS
BEV_KEY = 'bev_embed'
DECODER_OUTPUT_KEYS = ('all_cls_scores', 'all_bbox_preds')
FULL_OUTPUT_KEYS = DECODER_OUTPUT_KEYS + (BEV_KEY,)


def _ndarray(arr: np.ndarray, dev: tvm.runtime.Device):
    arr = np.ascontiguousarray(arr)
    if arr.dtype in (np.int64, np.int32, np.uint64, np.uint32):
        return tvm.nd.array(arr.astype(np.int64), dev)
    return tvm.nd.array(arr.astype(np.float32), dev)


def load_vm(so_path: str, dev_name: str = 'iluvatar', dev_id: int = 0):
    dev = tvm.device(dev_name, dev_id)
    return relax.VirtualMachine(tvm.runtime.load_module(so_path), dev), dev


def _to_numpy_list(outputs, n_expected: int | None = None) -> list[np.ndarray]:
    if isinstance(outputs, (list, tuple)):
        arrs = [x.numpy() for x in outputs]
    elif isinstance(outputs, tvm.ir.container.Array):
        arrs = [outputs[i].numpy() for i in range(len(outputs))]
    elif hasattr(outputs, 'numpy'):
        arrs = [outputs.numpy()]
    else:
        raise TypeError(f'unexpected VM output type: {type(outputs)}')
    if n_expected is not None and len(arrs) != n_expected:
        raise ValueError(f'expected {n_expected} outputs, got {len(arrs)}')
    return arrs


def run_backbone_igie(vm, dev, img_np: np.ndarray) -> dict[str, np.ndarray]:
    out = vm['main'](_ndarray(img_np, dev))
    feats = _to_numpy_list(out, len(FEAT_KEYS))
    return dict(zip(FEAT_KEYS, feats))


def run_backbone_probe_igie(vm, dev, img_np: np.ndarray, keys=None) -> dict[str, np.ndarray]:
    from export_utils import PROBE_KEYS

    probe_keys = keys or PROBE_KEYS
    out = vm['main'](_ndarray(img_np, dev))
    arrs = _to_numpy_list(out, len(probe_keys))
    return dict(zip(probe_keys, arrs))


def run_encoder_igie(
    vm,
    dev,
    feat_dict: dict[str, np.ndarray],
    lidar2img: np.ndarray,
    can_bus: np.ndarray,
    img_shape: np.ndarray,
) -> np.ndarray:
    inputs = [
        _ndarray(feat_dict[k], dev) for k in FEAT_KEYS
    ] + [
        _ndarray(lidar2img, dev),
        _ndarray(can_bus, dev),
        _ndarray(img_shape, dev),
    ]
    out = vm['main'](*inputs)
    arrs = _to_numpy_list(out, 1)
    return arrs[0]


def run_decoder_igie(vm, dev, bev_embed: np.ndarray) -> dict[str, np.ndarray]:
    out = vm['main'](_ndarray(bev_embed, dev))
    arrs = _to_numpy_list(out, len(DECODER_OUTPUT_KEYS))
    return dict(zip(DECODER_OUTPUT_KEYS, arrs))


def run_stacked_igie(
    backbone_vm,
    encoder_vm,
    decoder_vm,
    dev,
    img_np: np.ndarray,
    lidar2img: np.ndarray,
    can_bus: np.ndarray,
    img_shape: np.ndarray,
) -> dict[str, np.ndarray]:
    feats = run_backbone_igie(backbone_vm, dev, img_np)
    bev_embed = run_encoder_igie(
        encoder_vm, dev, feats, lidar2img, can_bus, img_shape)
    dec_out = run_decoder_igie(decoder_vm, dev, bev_embed)
    dec_out[BEV_KEY] = bev_embed
    return dec_out


def run_full_stack_igie(
    vm,
    dev,
    img_np: np.ndarray,
    lidar2img: np.ndarray,
    can_bus: np.ndarray,
    img_shape: np.ndarray,
) -> dict[str, np.ndarray]:
    inputs = [
        _ndarray(img_np, dev),
        _ndarray(lidar2img, dev),
        _ndarray(can_bus, dev),
        _ndarray(img_shape, dev),
    ]
    out = vm['main'](*inputs)
    arrs = _to_numpy_list(out, len(FULL_OUTPUT_KEYS))
    return dict(zip(FULL_OUTPUT_KEYS, arrs))


def compare_arrays(name: str, ref: np.ndarray, got: np.ndarray, atol: float) -> bool:
    diff = np.abs(ref.astype(np.float64) - got.astype(np.float64))
    corr = float(np.corrcoef(ref.ravel(), got.ravel())[0, 1]) if ref.size > 1 else 1.0
    ok = diff.max() <= atol
    print(
        f'  {name}: max_err={diff.max():.6e} mean_err={diff.mean():.6e} '
        f'corr={corr:.4f} -> {"PASS" if ok else "FAIL"}')
    return ok
