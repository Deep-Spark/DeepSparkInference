"""All implementations for the mmcv shim — imported by every sub-module."""

import functools
import logging
import math
import os
import os.path as osp
import warnings
from collections.abc import Mapping, Sequence

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# AttrDict / Config
# ---------------------------------------------------------------------------

class _AttrDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(name)


def _wrap_cfg(value):
    if isinstance(value, dict):
        ad = _AttrDict()
        for k, v in value.items():
            ad[k] = _wrap_cfg(v)
        return ad
    if isinstance(value, list):
        return [_wrap_cfg(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_wrap_cfg(v) for v in value)
    return value


class Config:
    def __init__(self, cfg_dict=None):
        object.__setattr__(self, '_cfg_dict', _wrap_cfg(cfg_dict or {}))

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, '_cfg_dict'), name)

    def __setattr__(self, name, value):
        object.__getattribute__(self, '_cfg_dict')[name] = _wrap_cfg(value)

    def __contains__(self, key):
        return key in object.__getattribute__(self, '_cfg_dict')

    @staticmethod
    def fromfile(path):
        path = osp.abspath(path)
        file_dirname = osp.dirname(path)
        with open(path, 'r') as f:
            code = f.read()
        # mmcv 1.x resolves {{fileDirname}} to the config file's directory
        code = code.replace('{{fileDirname}}', file_dirname.replace('\\', '/'))
        ns = {'__file__': path}
        exec(compile(code, path, 'exec'), ns)  # noqa: S102
        cfg_dict = {k: v for k, v in ns.items() if not k.startswith('_')}
        return Config(cfg_dict)


# ---------------------------------------------------------------------------
# FileClient
# ---------------------------------------------------------------------------

class FileClient:
    def __init__(self, backend='disk', **kwargs):
        self.backend = backend

    @classmethod
    def infer_client(cls, file_client_args, uri=None):
        return cls(**file_client_args)

    def get(self, filepath):
        with open(filepath, 'rb') as f:
            return f.read()

    def exists(self, filepath):
        return osp.exists(filepath)

    def isdir(self, filepath):
        return osp.isdir(filepath)

    def isfile(self, filepath):
        return osp.isfile(filepath)

    def list_dir_or_file(self, dir_path, list_dir=True, list_file=True,
                         suffix=None, recursive=False):
        if isinstance(suffix, str):
            suffix = (suffix,)
        if recursive:
            for root, dirs, files in os.walk(dir_path):
                rel_root = osp.relpath(root, dir_path)
                if list_dir:
                    for d in sorted(dirs):
                        rel = osp.join(rel_root, d) if rel_root != '.' else d
                        if suffix is None or any(rel.endswith(s) for s in suffix):
                            yield rel
                if list_file:
                    for fn in sorted(files):
                        rel = osp.join(rel_root, fn) if rel_root != '.' else fn
                        if suffix is None or any(rel.endswith(s) for s in suffix):
                            yield rel
        else:
            for entry in sorted(os.listdir(dir_path)):
                full = osp.join(dir_path, entry)
                is_dir = osp.isdir(full)
                is_file = osp.isfile(full)
                if is_dir and list_dir:
                    if suffix is None or any(entry.endswith(s) for s in suffix):
                        yield entry
                elif is_file and list_file:
                    if suffix is None or any(entry.endswith(s) for s in suffix):
                        yield entry


# ---------------------------------------------------------------------------
# File utilities
# ---------------------------------------------------------------------------

def mkdir_or_exist(dir_name, mode=0o777):
    os.makedirs(dir_name, mode=mode, exist_ok=True)


def scandir(dir_path, suffix=None, recursive=False):
    if isinstance(suffix, str):
        suffix = (suffix,)
    if recursive:
        for root, _, files in os.walk(dir_path):
            for fn in sorted(files):
                if suffix is None or any(fn.endswith(s) for s in suffix):
                    yield osp.relpath(osp.join(root, fn), dir_path)
    else:
        for fn in sorted(os.listdir(dir_path)):
            if osp.isfile(osp.join(dir_path, fn)):
                if suffix is None or any(fn.endswith(s) for s in suffix):
                    yield fn


def list_from_file(filename, prefix='', offset=0, max_num=None,
                   encoding='utf-8', file_client_args=None):
    with open(filename, encoding=encoding) as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines if x.strip()]
    if offset:
        lines = lines[offset:]
    if max_num is not None:
        lines = lines[:max_num]
    if prefix:
        lines = [prefix + line for line in lines]
    return lines


# ---------------------------------------------------------------------------
# Type-check helpers
# ---------------------------------------------------------------------------

def is_str(x):
    return isinstance(x, str)


def is_list_of(seq, expected_type):
    return isinstance(seq, list) and all(isinstance(i, expected_type) for i in seq)


def is_tuple_of(seq, expected_type):
    return isinstance(seq, tuple) and all(isinstance(i, expected_type) for i in seq)


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def get_logger(name, log_file=None, log_level=logging.INFO, file_mode='w'):
    logger = logging.getLogger(name)
    if not logger.handlers:
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        sh.setLevel(log_level)
        logger.addHandler(sh)
        if log_file is not None:
            fh = logging.FileHandler(log_file, mode=file_mode)
            fh.setLevel(log_level)
            logger.addHandler(fh)
    logger.setLevel(log_level)
    return logger


def print_log(msg, logger=None, level=logging.INFO):
    if logger is None or logger == 'print':
        print(msg)
    elif logger == 'silent':
        pass
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
    else:
        print(msg)


# ---------------------------------------------------------------------------
# Registry and build_from_cfg
# ---------------------------------------------------------------------------

class Registry:
    def __init__(self, name):
        self.name = name
        self._module_dict = {}

    def register_module(self, name=None, force=False, module=None):
        if module is not None:
            reg_name = name or module.__name__
            self._module_dict[reg_name] = module
            return module

        def decorator(cls):
            reg_name = name or cls.__name__
            self._module_dict[reg_name] = cls
            return cls

        return decorator

    def get(self, key):
        return self._module_dict.get(key)

    def build(self, cfg, default_args=None):
        return build_from_cfg(cfg, self, default_args)


def build_from_cfg(cfg, registry, default_args=None):
    import inspect
    if not isinstance(cfg, (dict, _AttrDict)):
        raise TypeError(f'cfg must be a dict, got {type(cfg)}')
    cfg = dict(cfg)
    obj_type = cfg.pop('type')
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(f"'{obj_type}' is not registered in '{registry.name}'")
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(f'type must be a str or class, got {type(obj_type)}')
    if default_args is not None:
        for k, v in default_args.items():
            cfg.setdefault(k, v)
    return obj_cls(**cfg)


# ---------------------------------------------------------------------------
# Version comparison
# ---------------------------------------------------------------------------

def digit_version(version_str):
    import re
    version_str = str(version_str).split('+')[0]
    parts = re.split(r'[^0-9]+', version_str)
    return tuple(int(p) for p in parts if p)


# ---------------------------------------------------------------------------
# Deprecation helper
# ---------------------------------------------------------------------------

def deprecated_api_warning(name_dict, cls_name=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for old_name, new_name in name_dict.items():
                if old_name in kwargs:
                    warnings.warn(
                        f"'{old_name}' is deprecated; use '{new_name}' instead.",
                        DeprecationWarning, stacklevel=2)
                    kwargs[new_name] = kwargs.pop(old_name)
            return func(*args, **kwargs)
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# DataContainer and collate (parallel module)
# ---------------------------------------------------------------------------

class DataContainer:
    """Minimal replacement for mmcv.parallel.DataContainer."""

    def __init__(self, data, stack=False, padding_value=0, cpu_only=False,
                 pad_dims=2):
        self._data = data
        self.stack = stack
        self.padding_value = padding_value
        self.cpu_only = cpu_only
        self.pad_dims = pad_dims

    @property
    def data(self):
        return self._data

    def __repr__(self):
        return f'{self.__class__.__name__}({self._data})'


def collate(batch, samples_per_gpu=1):
    """Batches a list of samples, handling DataContainer objects.

    Mirrors mmcv.parallel.collate.
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.data.dataloader import default_collate

    if not isinstance(batch, Sequence):
        raise TypeError(f'batch type {type(batch)} is not supported.')

    if isinstance(batch[0], DataContainer):
        stacked = []
        if batch[0].cpu_only:
            for i in range(0, len(batch), samples_per_gpu):
                stacked.append([s.data for s in batch[i:i + samples_per_gpu]])
            return DataContainer(stacked, batch[0].stack, batch[0].padding_value,
                                 cpu_only=True)
        elif batch[0].stack:
            for i in range(0, len(batch), samples_per_gpu):
                assert isinstance(batch[i].data, torch.Tensor)
                if batch[i].pad_dims is not None:
                    ndim = batch[i].data.dim()
                    assert ndim > batch[i].pad_dims
                    max_shape = [0] * batch[i].pad_dims
                    for dim in range(1, batch[i].pad_dims + 1):
                        max_shape[dim - 1] = batch[i].data.shape[-dim]
                    for sample in batch[i:i + samples_per_gpu]:
                        for dim in range(batch[i].pad_dims):
                            max_shape[dim] = max(max_shape[dim],
                                                 sample.data.shape[-(dim + 1)])
                    padded = []
                    for sample in batch[i:i + samples_per_gpu]:
                        pad = [0] * (batch[i].pad_dims * 2)
                        for dim in range(1, batch[i].pad_dims + 1):
                            pad[2 * dim - 2] = (max_shape[dim - 1]
                                                - sample.data.shape[-dim])
                        padded.append(F.pad(sample.data, pad,
                                            value=sample.padding_value))
                    stacked.append(default_collate(padded))
                else:
                    stacked.append(default_collate(
                        [s.data for s in batch[i:i + samples_per_gpu]]))
            return DataContainer(stacked, batch[0].stack, batch[0].padding_value)
        else:
            for i in range(0, len(batch), samples_per_gpu):
                stacked.append([s.data for s in batch[i:i + samples_per_gpu]])
            return DataContainer(stacked, batch[0].stack, batch[0].padding_value)

    elif isinstance(batch[0], torch.Tensor):
        return default_collate(batch)
    elif isinstance(batch[0], str):
        return batch
    elif isinstance(batch[0], Sequence):
        transposed = list(zip(*batch))
        return [collate(samples, samples_per_gpu) for samples in transposed]
    elif isinstance(batch[0], Mapping):
        return {key: collate([d[key] for d in batch], samples_per_gpu)
                for key in batch[0]}
    else:
        from torch.utils.data.dataloader import default_collate
        return default_collate(batch)


# ---------------------------------------------------------------------------
# Runner utilities
# ---------------------------------------------------------------------------

def get_dist_info():
    """Always returns (rank=0, world_size=1) for single-process inference."""
    return 0, 1


# ---------------------------------------------------------------------------
# Interpolation map
# ---------------------------------------------------------------------------

_INTERP_CODES = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4,
}


def _interp_flag(interpolation):
    return _INTERP_CODES.get(interpolation, cv2.INTER_LINEAR)


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

_FLAG_MAP = {
    'color': cv2.IMREAD_COLOR,
    'grayscale': cv2.IMREAD_GRAYSCALE,
    'unchanged': cv2.IMREAD_UNCHANGED,
}


def imfrombytes(content, flag='color', channel_order='bgr', backend=None):
    buf = np.frombuffer(content, dtype=np.uint8)
    imread_flag = _FLAG_MAP.get(flag, cv2.IMREAD_COLOR)
    img = cv2.imdecode(buf, imread_flag)
    if img is None:
        raise OSError('Failed to decode image from bytes')
    if channel_order == 'rgb' and img.ndim == 3:
        img = img[:, :, ::-1].copy()
    return img


def imread(img_or_path, flag='color', channel_order='bgr', backend=None):
    if isinstance(img_or_path, np.ndarray):
        return img_or_path
    imread_flag = _FLAG_MAP.get(flag, cv2.IMREAD_COLOR)
    img = cv2.imread(img_or_path, imread_flag)
    if img is None:
        raise FileNotFoundError(f'Image not found or unreadable: {img_or_path}')
    if channel_order == 'rgb' and img.ndim == 3:
        img = img[:, :, ::-1].copy()
    return img


# ---------------------------------------------------------------------------
# Image resize
# ---------------------------------------------------------------------------

def imresize(img, size, return_scale=False, interpolation='bilinear',
             backend=None, **kwargs):
    h, w = img.shape[:2]
    new_w, new_h = int(size[0]), int(size[1])
    resized = cv2.resize(img, (new_w, new_h),
                         interpolation=_interp_flag(interpolation))
    if not return_scale:
        return resized
    return resized, new_w / w, new_h / h


def _rescale_size(old_size, scale, return_scale=False):
    w, h = old_size
    if isinstance(scale, (float, int)):
        scale_factor = float(scale)
    elif isinstance(scale, tuple):
        max_long = max(scale)
        max_short = min(scale)
        long_edge = max(h, w)
        short_edge = min(h, w)
        scale_factor = min(max_long / long_edge, max_short / short_edge)
    else:
        raise TypeError(f'Scale must be float or tuple, got {type(scale)}')
    new_w = int(w * scale_factor + 0.5)
    new_h = int(h * scale_factor + 0.5)
    if return_scale:
        return (new_w, new_h), scale_factor
    return (new_w, new_h)


def imrescale(img, scale, return_scale=False, interpolation='bilinear',
              backend=None):
    h, w = img.shape[:2]
    new_size, scale_factor = _rescale_size((w, h), scale, return_scale=True)
    resized = cv2.resize(img, new_size, interpolation=_interp_flag(interpolation))
    if return_scale:
        return resized, scale_factor
    return resized


def imresize_to_multiple(img, divisor, scale_factor=1, interpolation='bilinear',
                         backend=None):
    h, w = img.shape[:2]
    new_h = math.ceil(h * scale_factor / divisor) * divisor
    new_w = math.ceil(w * scale_factor / divisor) * divisor
    return cv2.resize(img, (new_w, new_h), interpolation=_interp_flag(interpolation))


# ---------------------------------------------------------------------------
# Image padding
# ---------------------------------------------------------------------------

def impad(img, *, shape=None, padding=None, pad_val=0, padding_mode='constant'):
    if shape is not None:
        target_h, target_w = int(shape[0]), int(shape[1])
        h, w = img.shape[:2]
        pad_h = max(target_h - h, 0)
        pad_w = max(target_w - w, 0)
        pw = ((0, pad_h), (0, pad_w)) if img.ndim == 2 else ((0, pad_h), (0, pad_w), (0, 0))
    elif padding is not None:
        top, bottom, left, right = padding
        pw = ((top, bottom), (left, right)) if img.ndim == 2 else \
             ((top, bottom), (left, right), (0, 0))
    else:
        raise ValueError('Either shape or padding must be provided')
    if padding_mode == 'constant':
        return np.pad(img, pw, mode='constant', constant_values=pad_val)
    return np.pad(img, pw, mode=padding_mode)


def impad_to_multiple(img, divisor, pad_val=0):
    h, w = img.shape[:2]
    new_h = math.ceil(h / divisor) * divisor
    new_w = math.ceil(w / divisor) * divisor
    return impad(img, shape=(new_h, new_w), pad_val=pad_val)


# ---------------------------------------------------------------------------
# Image flip
# ---------------------------------------------------------------------------

_FLIP_CODES = {'horizontal': 1, 'vertical': 0, 'diagonal': -1}


def imflip(img, direction='horizontal'):
    code = _FLIP_CODES.get(direction)
    if code is None:
        raise ValueError(f"Invalid flip direction: '{direction}'")
    return cv2.flip(img, code)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def imnormalize(img, mean, std, to_rgb=True):
    img = img.astype(np.float32)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    if to_rgb and img.ndim == 3 and img.shape[2] == 3:
        img = img[:, :, ::-1]
    img = (img - mean) / std
    return img.astype(np.float32)


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------

def imrotate(img, angle, center=None, scale=1.0, border_value=0,
             auto_bound=False, interpolation='bilinear'):
    h, w = img.shape[:2]
    if center is None:
        center = ((w - 1) / 2.0, (h - 1) / 2.0)
    M = cv2.getRotationMatrix2D(center, -angle, scale)
    if auto_bound:
        cos = abs(M[0, 0])
        sin = abs(M[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2
        w, h = new_w, new_h
    flags = _interp_flag(interpolation)
    return cv2.warpAffine(img, M, (w, h), flags=flags, borderValue=border_value)


# ---------------------------------------------------------------------------
# Color / LUT
# ---------------------------------------------------------------------------

def clahe(img, clip_limit=40.0, tile_grid_size=(8, 8)):
    obj = cv2.createCLAHE(clipLimit=float(clip_limit),
                           tileGridSize=tuple(tile_grid_size))
    return obj.apply(img)


def bgr2hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def hsv2bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)


def lut_transform(img, lut_table):
    return cv2.LUT(img, lut_table)
