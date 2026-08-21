"""Shared helpers for BEVFormer pt2 export / IGIE deployment."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch
from mmcv import Config
from mmcv.runner import load_checkpoint

REPO_ROOT = Path(__file__).resolve().parents[2]


def ensure_repo_on_path() -> Path:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    return REPO_ROOT


def remove_runtime_assertions(exported_program):
    """Drop export-time guard ops that block downstream compilers (e.g. TVM)."""
    from torch._export.passes.remove_runtime_assertions import _RemoveRuntimeAssertionsPass

    result = _RemoveRuntimeAssertionsPass()(exported_program.graph_module)
    if result.modified:
        exported_program.graph_module.recompile()
        print('Removed runtime assertion nodes from exported graph')
    return exported_program


def install_frozen_bn_export() -> None:
    """Export eval BN as explicit elementwise ops (bypass IGIE relax.nn.batch_norm)."""
    import torch.nn as nn

    if getattr(nn.BatchNorm2d, '_igie_frozen_export_patched', False):
        return

    _orig_bn2d = nn.BatchNorm2d.forward

    def _bn2d_forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fold eval BN to elementwise ops in both eager eval and torch.export.
        if not self.training:
            c = self.num_features
            rm = self.running_mean.reshape(1, c, 1, 1).to(dtype=x.dtype)
            rv = self.running_var.reshape(1, c, 1, 1).to(dtype=x.dtype)
            w = self.weight.reshape(1, c, 1, 1).to(dtype=x.dtype)
            b = self.bias.reshape(1, c, 1, 1).to(dtype=x.dtype)
            return (x - rm) * (w * torch.rsqrt(rv + self.eps)) + b
        return _orig_bn2d(self, x)

    nn.BatchNorm2d.forward = _bn2d_forward
    nn.BatchNorm2d._igie_frozen_export_patched = True
    print('installed frozen BatchNorm2d export (elementwise, no batch_norm op)')


def count_batch_norm_ops(exported_program) -> int:
    n = 0
    for node in exported_program.graph_module.graph.nodes:
        if node.op == 'call_function' and 'batch_norm' in str(node.target):
            n += 1
    return n


def import_plugin_modules(cfg, config_path: str) -> None:
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    if not getattr(cfg, 'plugin', False):
        return

    if hasattr(cfg, 'plugin_dir'):
        module_path = Path(cfg.plugin_dir)
    else:
        module_path = Path(config_path).resolve().parent

    if not module_path.is_absolute():
        module_path = (REPO_ROOT / module_path).resolve()

    try:
        relative_path = module_path.relative_to(REPO_ROOT)
    except ValueError:
        relative_path = module_path

    importlib.import_module('.'.join(relative_path.parts))


def build_bevformer_model(config_path: str, checkpoint_path: str, device: str):
    ensure_repo_on_path()
    from mmdet3d.models import build_model

    cfg = Config.fromfile(config_path)
    import_plugin_modules(cfg, config_path)

    cfg.model.pretrained = None
    cfg.model.train_cfg = None

    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, checkpoint_path, map_location='cpu')
    model.to(device)
    model.eval()
    return model, cfg


def bev_ms_deform_shapes(cfg, mode: str):
    """Return (num_query, value_len, spatial_shapes_list) for isolated ms_deform export."""
    bev_h = cfg.model['pts_bbox_head']['bev_h']
    bev_w = cfg.model['pts_bbox_head']['bev_w']
    embed_dims = cfg.model['pts_bbox_head']['transformer']['embed_dims']
    num_query = cfg.model['pts_bbox_head']['num_query']

    if mode == 'decoder':
        # CustomMSDeformableAttention: num_levels=1, attends to BEV grid
        return num_query, bev_h * bev_w, [(bev_h, bev_w)], embed_dims

    if mode == 'tsa':
        # TemporalSelfAttention encoder self-attn: num_levels=1 on BEV grid
        return bev_h * bev_w, bev_h * bev_w, [(bev_h, bev_w)], embed_dims

    raise ValueError(f'unknown mode: {mode}')


def bev_sca_ms_deform_shapes(
    cfg,
    image_shape: tuple[int, int, int, int, int] | None = None,
):
    """Return shapes for isolated MSDeformableAttention3D (SCA 4-level path).

    Returns:
        num_query, value_len, spatial_shapes, embed_dims, num_z_anchors
    """
    bev_h = cfg.model['pts_bbox_head']['bev_h']
    bev_w = cfg.model['pts_bbox_head']['bev_w']
    embed_dims = cfg.model['pts_bbox_head']['transformer']['embed_dims']
    num_query = bev_h * bev_w
    num_z_anchors = cfg.model['pts_bbox_head']['transformer']['encoder']['num_points_in_pillar']

    if image_shape is None:
        image_shape = (1, 6, 3, 928, 1600)
    batch_size, num_cams, _, img_h, img_w = image_shape
    feat_shapes = fpn_feat_shapes_from_image(batch_size, num_cams, img_h, img_w)
    spatial_shapes = [(int(s[2]), int(s[3])) for s in feat_shapes]
    value_len = sum(h * w for h, w in spatial_shapes)
    return num_query, value_len, spatial_shapes, embed_dims, num_z_anchors


def extract_image_features(model, img: torch.Tensor):
    """img [B, N, 3, H, W] -> tuple of FPN feats, each [B*N, C, h, w]."""
    batch_size, num_cams, channels, height, width = img.shape
    x = img.reshape(batch_size * num_cams, channels, height, width)

    feats = model.img_backbone(x)
    if isinstance(feats, dict):
        feats = list(feats.values())
    if model.with_img_neck:
        feats = model.img_neck(feats)
    return tuple(feats)


def infer_fpn_feat_shapes(model, image_shape: tuple[int, ...], device: str):
    """Run a dummy forward to get FPN output shapes for fixed image_shape."""
    dummy = torch.randn(*image_shape, dtype=torch.float32, device=device)
    with torch.no_grad():
        feats = extract_image_features(model, dummy)
    return [tuple(f.shape) for f in feats]


def fpn_feat_shapes_from_image(
    batch_size: int,
    num_cams: int,
    img_h: int,
    img_w: int,
    channels: int = 256,
) -> list[tuple[int, int, int, int]]:
    """Static FPN shapes for bevformer-base (ResNet101 out_indices 1,2,3 + FPN extra level)."""
    bn = batch_size * num_cams
    strides = (8, 16, 32, 64)
    return [
        (bn, channels, (img_h + s - 1) // s, (img_w + s - 1) // s)
        for s in strides
    ]


def feat_shapes_from_exported_program(
    exported_program,
    feat_names: tuple[str, ...] = ('feat0', 'feat1', 'feat2', 'feat3'),
) -> list[tuple[int, ...]]:
    """Read FPN feat input shapes from torch.export user inputs (not weights)."""
    import torch.export.graph_signature as graph_signature

    def _placeholder_shape(target: str, name_hint: str) -> tuple[int, ...]:
        for node in exported_program.graph.find_nodes(op='placeholder', target=target):
            if node.name != name_hint:
                continue
            meta = node.meta
            if 'tensor_meta' in meta:
                return tuple(int(s) for s in meta['tensor_meta'].shape)
            val = meta.get('val')
            if val is not None and hasattr(val, 'shape'):
                return tuple(int(s) for s in val.shape)
            if 'grapharg' in meta and meta['grapharg'].fake_tensor is not None:
                ft = meta['grapharg'].fake_tensor
                return tuple(int(s) for s in ft.shape)
        raise ValueError(
            f'cannot read shape for user input {name_hint!r} (target={target!r})')

    by_name: dict[str, tuple[int, ...]] = {}
    for spec in exported_program.graph_signature.input_specs:
        if spec.kind is not graph_signature.InputKind.USER_INPUT:
            continue
        by_name[spec.arg.name] = _placeholder_shape(spec.target, spec.arg.name)

    missing = [n for n in feat_names if n not in by_name]
    if missing:
        raise ValueError(
            f'missing feat user inputs in pt2: {missing}; '
            f'available user inputs: {sorted(by_name)}')
    return [by_name[n] for n in feat_names]


def patch_model_for_pt2_export(model) -> None:
    """Tensorize img_metas paths used by BEV head (same as ONNX export)."""
    import math
    import types

    def point_sampling_tensor(self, reference_points, pc_range, img_metas):
        lidar2img = torch.stack([meta['lidar2img'] for meta in img_metas], dim=0)
        img_shape = torch.stack([meta['img_shape'] for meta in img_metas], dim=0)

        reference_points = reference_points.clone()
        # Avoid in-place strided slice writes; IGIE lowers them as no-ops on x/y.
        x = reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        y = reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        z = reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
        reference_points = torch.cat([x, y, z], dim=-1)

        reference_points = torch.cat(
            [reference_points, torch.ones_like(reference_points[..., :1])], dim=-1)

        reference_points = reference_points.permute(1, 0, 2, 3)
        depth_bins, batch_size, num_query = reference_points.shape[:3]
        num_cams = lidar2img.shape[1]

        reference_points = reference_points.view(
            depth_bins, batch_size, 1, num_query, 4, 1).repeat(
                1, 1, num_cams, 1, 1, 1)
        lidar2img = lidar2img.view(
            1, batch_size, num_cams, 1, 4, 4).repeat(
                depth_bins, 1, 1, num_query, 1, 1)

        reference_points_cam = torch.matmul(
            lidar2img.float(), reference_points.float()).squeeze(-1)
        eps = 1e-5
        mask_dtype = reference_points_cam.dtype

        bev_mask = (reference_points_cam[..., 2:3] > eps).to(dtype=mask_dtype)
        reference_points_cam = reference_points_cam[..., 0:2] / torch.clamp(
            reference_points_cam[..., 2:3], min=eps)

        img_h = img_shape[..., 0].view(1, batch_size, num_cams, 1, 1)
        img_w = img_shape[..., 1].view(1, batch_size, num_cams, 1, 1)
        u = reference_points_cam[..., 0:1] / img_w
        v = reference_points_cam[..., 1:2] / img_h
        reference_points_cam = torch.cat([u, v], dim=-1)

        bev_mask = bev_mask * (reference_points_cam[..., 1:2] > 0.0).to(dtype=mask_dtype)
        bev_mask = bev_mask * (reference_points_cam[..., 1:2] < 1.0).to(dtype=mask_dtype)
        bev_mask = bev_mask * (reference_points_cam[..., 0:1] > 0.0).to(dtype=mask_dtype)
        bev_mask = bev_mask * (reference_points_cam[..., 0:1] < 1.0).to(dtype=mask_dtype)

        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)
        return reference_points_cam, bev_mask

    def get_bev_features_tensor(self,
                                mlvl_feats,
                                bev_queries,
                                bev_h,
                                bev_w,
                                grid_length=(0.512, 0.512),
                                bev_pos=None,
                                prev_bev=None,
                                **kwargs):
        if prev_bev is not None:
            raise NotImplementedError(
                'This export path supports only single-frame inference.')

        img_metas = kwargs['img_metas']
        batch_size = mlvl_feats[0].size(0)

        bev_queries = bev_queries.unsqueeze(1).repeat(1, batch_size, 1)
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)

        can_bus = torch.stack([meta['can_bus'] for meta in img_metas], dim=0)
        can_bus = can_bus.to(dtype=bev_queries.dtype)

        delta_x = can_bus[:, 0]
        delta_y = can_bus[:, 1]
        # Use positive index: can_bus is [B, 18]; -2 → 16 (avoid igie take(-1) OOB).
        ego_angle = can_bus[:, 16] / math.pi * 180.0

        grid_length_y = grid_length[0]
        grid_length_x = grid_length[1]
        translation_length = torch.sqrt(delta_x ** 2 + delta_y ** 2)
        translation_angle = torch.atan2(delta_y, delta_x) / math.pi * 180.0
        bev_angle = ego_angle - translation_angle

        shift_y = translation_length * torch.cos(
            bev_angle / 180.0 * math.pi) / grid_length_y / bev_h
        shift_x = translation_length * torch.sin(
            bev_angle / 180.0 * math.pi) / grid_length_x / bev_w
        if self.use_shift:
            shift = torch.stack([shift_x, shift_y], dim=-1)
        else:
            shift = torch.zeros(
                batch_size, 2, dtype=bev_queries.dtype, device=bev_queries.device)

        can_bus_embed = self.can_bus_mlp(can_bus)[None, :, :]
        if self.use_can_bus:
            bev_queries = bev_queries + can_bus_embed

        feat_flatten = []
        spatial_shapes_list = []
        for level, feat in enumerate(mlvl_feats):
            batch_size, num_cams, channels, feat_h, feat_w = feat.shape
            spatial_shapes_list.append((feat_h, feat_w))

            feat = feat.flatten(3).permute(1, 0, 3, 2)
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[
                None, None, level:level + 1, :].to(feat.dtype)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, dim=2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes_list, dtype=torch.long, device=bev_pos.device)
        # Precompute level starts as Python ints (avoid cumsum[:-1] → igie slice -1).
        level_starts = [0]
        for h, w in spatial_shapes_list[:-1]:
            level_starts.append(level_starts[-1] + int(h) * int(w))
        level_start_index = torch.tensor(
            level_starts, dtype=torch.long, device=bev_pos.device)
        feat_flatten = feat_flatten.permute(0, 2, 1, 3)

        return self.encoder(
            bev_queries,
            feat_flatten,
            feat_flatten,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            prev_bev=None,
            shift=shift,
            img_metas=img_metas,
            **{k: v for k, v in kwargs.items() if k not in ('img_metas',)},
        )

    transformer = model.pts_bbox_head.transformer
    encoder = transformer.encoder
    transformer.get_bev_features = types.MethodType(
        get_bev_features_tensor, transformer)
    encoder.point_sampling = types.MethodType(
        point_sampling_tensor, encoder)


def patch_decoder_for_pt2_export(model) -> None:
    """Avoid in-place strided slice writes in decoder reference refine (IGIE no-op bug)."""
    from projects.mmdet3d_plugin.bevformer.modules.decoder import (
        DetectionTransformerDecoder,
        inverse_sigmoid,
    )

    if getattr(DetectionTransformerDecoder, '_igie_slice_patch', False):
        return

    def _decoder_forward(self, query, *args, reference_points=None, reg_branches=None,
                         key_padding_mask=None, **kwargs):
        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points[..., :2].unsqueeze(2)
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                key_padding_mask=key_padding_mask,
                **kwargs)
            output = output.permute(1, 0, 2)

            if reg_branches is not None:
                tmp = reg_branches[lid](output)
                xy = tmp[..., :2] + inverse_sigmoid(reference_points[..., :2])
                z = tmp[..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])
                reference_points = torch.cat([xy, z], dim=-1).sigmoid().detach()

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)
        return output, reference_points

    DetectionTransformerDecoder.forward = _decoder_forward
    DetectionTransformerDecoder._igie_slice_patch = True


def patch_head_bbox_for_pt2_export(model) -> None:
    """Avoid in-place slice writes in BEVFormerHead cls/reg output loop."""
    from projects.mmdet3d_plugin.bevformer.dense_heads.bevformer_head import BEVFormerHead

    if getattr(BEVFormerHead, '_igie_bbox_slice_patch', False):
        return

    def _forward(self, mlvl_feats, img_metas, prev_bev=None, only_bev=False):
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        object_query_embeds = self.query_embedding.weight.to(dtype)
        bev_queries = self.bev_embedding.weight.to(dtype)

        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)

        if only_bev:
            return self.transformer.get_bev_features(
                mlvl_feats,
                bev_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )

        outputs = self.transformer(
            mlvl_feats,
            bev_queries,
            object_query_embeds,
            self.bev_h,
            self.bev_w,
            grid_length=(self.real_h / self.bev_h,
                         self.real_w / self.bev_w),
            bev_pos=bev_pos,
            reg_branches=self.reg_branches if self.with_box_refine else None,
            cls_branches=self.cls_branches if self.as_two_stage else None,
            img_metas=img_metas,
            prev_bev=prev_bev,
        )

        bev_embed, hs, init_reference, inter_references = outputs
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []
        pc = self.pc_range
        for lvl in range(hs.shape[0]):
            reference = init_reference if lvl == 0 else inter_references[lvl - 1]
            cls_out, bbox_out = _decoder_bbox_outputs(
                self, hs[lvl], reference, init_reference, lvl, pc)
            outputs_classes.append(cls_out)
            outputs_coords.append(bbox_out)

        return {
            'bev_embed': bev_embed,
            'all_cls_scores': torch.stack(outputs_classes),
            'all_bbox_preds': torch.stack(outputs_coords),
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
        }

    BEVFormerHead.forward = _forward
    BEVFormerHead._igie_bbox_slice_patch = True


def _decoder_bbox_outputs(head, hs_lvl, reference, init_reference, lvl: int, pc):
    """Cls/reg head outputs without in-place slice writes."""
    from projects.mmdet3d_plugin.bevformer.modules.decoder import inverse_sigmoid

    reference = init_reference if lvl == 0 else reference
    reference = inverse_sigmoid(reference)
    outputs_class = head.cls_branches[lvl](hs_lvl)
    tmp = head.reg_branches[lvl](hs_lvl)

    xy_sig = (tmp[..., 0:2] + reference[..., 0:2]).sigmoid()
    z_sig = (tmp[..., 4:5] + reference[..., 2:3]).sigmoid()
    tmp = torch.cat([
        xy_sig[..., 0:1], xy_sig[..., 1:2],
        tmp[..., 2:4],
        z_sig,
        tmp[..., 5:],
    ], dim=-1)

    x = tmp[..., 0:1] * (pc[3] - pc[0]) + pc[0]
    y = tmp[..., 1:2] * (pc[4] - pc[1]) + pc[1]
    z = tmp[..., 4:5] * (pc[5] - pc[2]) + pc[2]
    tmp = torch.cat([
        x, y,
        tmp[..., 2:4],
        z,
        tmp[..., 5:],
    ], dim=-1)
    return outputs_class, tmp


def patch_encoder_early_exit(model, num_layers: int | None) -> None:
    """Stop encoder after num_layers without mutating ModuleList (export-friendly)."""
    if num_layers is None:
        return
    encoder = model.pts_bbox_head.transformer.encoder
    if getattr(encoder, '_igie_early_exit_layers', None) == num_layers:
        return

    _orig_forward = encoder.forward

    def _forward(self, *args, **kwargs):
        output = _orig_forward(*args, **kwargs)
        return output

    # Reimplement loop with early exit by patching forward body via wrapper.
    import types
    from projects.mmdet3d_plugin.bevformer.modules.encoder import BEVFormerEncoder

    _orig_enc_forward = BEVFormerEncoder.forward

    def _enc_forward(self, bev_query, key, value, *args, **kwargs):
        if getattr(self, '_igie_early_exit_layers', None) is None:
            return _orig_enc_forward(self, bev_query, key, value, *args, **kwargs)

        max_layers = self._igie_early_exit_layers
        output = bev_query
        intermediate = []

        ref_3d = self.get_reference_points(
            kwargs.get('bev_h'), kwargs.get('bev_w'),
            self.pc_range[5] - self.pc_range[2], self.num_points_in_pillar,
            dim='3d', bs=bev_query.size(1), device=bev_query.device,
            dtype=bev_query.dtype)
        ref_2d = self.get_reference_points(
            kwargs.get('bev_h'), kwargs.get('bev_w'),
            dim='2d', bs=bev_query.size(1), device=bev_query.device,
            dtype=bev_query.dtype)

        reference_points_cam, bev_mask = self.point_sampling(
            ref_3d, self.pc_range, kwargs['img_metas'])

        shift = kwargs.get('shift', 0.)
        shift_ref_2d = ref_2d.clone()
        shift_ref_2d = shift_ref_2d + shift[:, None, None, :]

        bev_query = bev_query.permute(1, 0, 2)
        bev_pos = kwargs.get('bev_pos')
        if bev_pos is not None:
            bev_pos = bev_pos.permute(1, 0, 2)
        bs, len_bev, num_bev_level, _ = ref_2d.shape
        prev_bev = kwargs.get('prev_bev')
        if prev_bev is not None:
            prev_bev = prev_bev.permute(1, 0, 2)
            prev_bev = torch.stack([prev_bev, bev_query], 1).reshape(bs * 2, len_bev, -1)
            hybird_ref_2d = torch.stack([shift_ref_2d, ref_2d], 1).reshape(
                bs * 2, len_bev, num_bev_level, 2)
        else:
            hybird_ref_2d = torch.stack([ref_2d, ref_2d], 1).reshape(
                bs * 2, len_bev, num_bev_level, 2)

        for lid, layer in enumerate(self.layers):
            if lid >= max_layers:
                break
            output = layer(
                bev_query,
                key,
                value,
                *args,
                bev_pos=bev_pos,
                ref_2d=hybird_ref_2d,
                ref_3d=ref_3d,
                bev_h=kwargs.get('bev_h'),
                bev_w=kwargs.get('bev_w'),
                spatial_shapes=kwargs.get('spatial_shapes'),
                level_start_index=kwargs.get('level_start_index'),
                reference_points_cam=reference_points_cam,
                bev_mask=bev_mask,
                prev_bev=prev_bev,
            )
            bev_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)
        return output

    BEVFormerEncoder.forward = _enc_forward
    encoder._igie_early_exit_layers = num_layers
    print(f'patched encoder early exit after {num_layers} layer(s)')


class BackboneNeckWrapper(torch.nn.Module):
    """img [B, N, 3, H, W] -> feat0..feat3, each [B*N, C, h, w]."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, img):
        return extract_image_features(self.model, img)


PROBE_KEYS = (
    'stem', 'l1', 'l2', 'l3', 'l4',
    'bb0', 'bb1', 'bb2',
    'feat0', 'feat1', 'feat2', 'feat3',
)


class BackboneProbeWrapper(torch.nn.Module):
    """img -> intermediate ResNet/FPN probes + final feat0..feat3."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, img):
        batch_size, num_cams, channels, height, width = img.shape
        x = img.reshape(batch_size * num_cams, channels, height, width)

        bb = self.model.img_backbone
        x = bb.conv1(x)
        x = bb.bn1(x)
        x = bb.relu(x)
        stem = x
        x = bb.maxpool(x)

        layer_outs = {}
        for i, layer_name in enumerate(bb.res_layers):
            x = getattr(bb, layer_name)(x)
            layer_outs[f'l{i + 1}'] = x

        bb_feats = []
        for i, layer_name in enumerate(bb.res_layers):
            if i in bb.out_indices:
                bb_feats.append(layer_outs[f'l{i + 1}'])

        if self.model.with_img_neck:
            neck_feats = self.model.img_neck(bb_feats)
            if isinstance(neck_feats, dict):
                neck_feats = list(neck_feats.values())
        else:
            neck_feats = bb_feats

        return (
            stem,
            layer_outs['l1'],
            layer_outs['l2'],
            layer_outs['l3'],
            layer_outs['l4'],
            bb_feats[0],
            bb_feats[1],
            bb_feats[2],
            neck_feats[0],
            neck_feats[1],
            neck_feats[2],
            neck_feats[3],
        )


class HeadWrapper(torch.nn.Module):
    """BEV head with tensorized img_metas (single-frame, no prev_bev)."""

    def __init__(self, model, batch_size: int, num_cams: int):
        super().__init__()
        self.model = model
        self.batch_size = batch_size
        self.num_cams = num_cams

    def _build_inputs(self, feat0, feat1, feat2, feat3, lidar2img, can_bus, img_shape):
        mlvl_feats = []
        for feat in (feat0, feat1, feat2, feat3):
            mlvl_feats.append(
                feat.reshape(
                    self.batch_size,
                    self.num_cams,
                    feat.shape[1],
                    feat.shape[2],
                    feat.shape[3]))

        img_metas = []
        for batch_idx in range(self.batch_size):
            img_metas.append({
                'lidar2img': lidar2img[batch_idx],
                'can_bus': can_bus[batch_idx],
                'img_shape': img_shape[batch_idx],
            })
        return mlvl_feats, img_metas

    def forward(self, feat0, feat1, feat2, feat3, lidar2img, can_bus, img_shape):
        mlvl_feats, img_metas = self._build_inputs(
            feat0, feat1, feat2, feat3, lidar2img, can_bus, img_shape)

        outputs = self.model.pts_bbox_head(
            mlvl_feats,
            img_metas,
            prev_bev=None)

        return (
            outputs['all_cls_scores'],
            outputs['all_bbox_preds'],
            outputs['bev_embed'],
        )


class EncoderWrapper(torch.nn.Module):
    """Encoder-only path: same inputs as HeadWrapper, returns bev_embed only."""

    def __init__(self, model, batch_size: int, num_cams: int):
        super().__init__()
        self.model = model
        self.batch_size = batch_size
        self.num_cams = num_cams

    def forward(self, feat0, feat1, feat2, feat3, lidar2img, can_bus, img_shape):
        head = HeadWrapper(self.model, self.batch_size, self.num_cams)
        mlvl_feats, img_metas = head._build_inputs(
            feat0, feat1, feat2, feat3, lidar2img, can_bus, img_shape)

        bev_embed = self.model.pts_bbox_head(
            mlvl_feats,
            img_metas,
            prev_bev=None,
            only_bev=True)
        return bev_embed.permute(1, 0, 2)


class DecoderWrapper(torch.nn.Module):
    """Decoder + cls/reg heads from encoder bev_embed (len_bev, bs, dim)."""

    def __init__(self, model, batch_size: int):
        super().__init__()
        self.model = model
        self.batch_size = batch_size
        self.bev_h = model.pts_bbox_head.bev_h
        self.bev_w = model.pts_bbox_head.bev_w
        self.register_buffer(
            'spatial_shapes',
            torch.tensor([[self.bev_h, self.bev_w]], dtype=torch.long))
        self.register_buffer(
            'level_start_index',
            torch.tensor([0], dtype=torch.long))

    def forward(self, bev_embed):
        head = self.model.pts_bbox_head
        transformer = head.transformer
        bs = self.batch_size
        dtype = bev_embed.dtype
        object_query_embeds = head.query_embedding.weight.to(dtype)

        query_pos, query = torch.split(
            object_query_embeds, transformer.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        init_reference = transformer.reference_points(query_pos).sigmoid()

        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)

        inter_states, inter_references = transformer.decoder(
            query=query,
            key=None,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=init_reference,
            reg_branches=head.reg_branches if head.with_box_refine else None,
            cls_branches=head.cls_branches if head.as_two_stage else None,
            spatial_shapes=self.spatial_shapes.to(device=bev_embed.device),
            level_start_index=self.level_start_index.to(device=bev_embed.device),
        )

        hs = inter_states.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []
        pc = head.pc_range
        for lvl in range(hs.shape[0]):
            cls_out, bbox_out = _decoder_bbox_outputs(
                head, hs[lvl], inter_references[lvl - 1], init_reference, lvl, pc)
            outputs_classes.append(cls_out)
            outputs_coords.append(bbox_out)

        return torch.stack(outputs_classes), torch.stack(outputs_coords)


class FullStackWrapper(torch.nn.Module):
    """img + img_metas -> cls/bbox/bev_embed (backbone+neck+encoder+decoder)."""

    def __init__(self, model, batch_size: int, num_cams: int):
        super().__init__()
        self.model = model
        self.head = HeadWrapper(model, batch_size, num_cams)

    def forward(self, img, lidar2img, can_bus, img_shape):
        feats = extract_image_features(self.model, img)
        return self.head(*feats, lidar2img, can_bus, img_shape)


def make_full_stack_dummy_inputs(
    batch_size: int,
    num_cams: int,
    img_h: int,
    img_w: int,
    device: str,
):
    """Build random img + img_metas for full-stack export/compare."""
    img = torch.randn(
        batch_size, num_cams, 3, img_h, img_w,
        dtype=torch.float32, device=device)
    lidar2img = torch.eye(
        4, dtype=torch.float32, device=device).view(1, 1, 4, 4).repeat(
            batch_size, num_cams, 1, 1)
    can_bus = torch.zeros(
        batch_size, 18, dtype=torch.float32, device=device)
    img_shape = torch.tensor(
        [img_h, img_w], dtype=torch.float32, device=device).view(
            1, 1, 2).repeat(batch_size, num_cams, 1)
    return img, lidar2img, can_bus, img_shape


def make_head_dummy_inputs(
    batch_size: int,
    num_cams: int,
    img_h: int,
    img_w: int,
    feat_shapes: list[tuple[int, ...]],
    device: str,
):
    """Build random head inputs matching export shapes."""
    feats = []
    for shape in feat_shapes:
        _, channels, feat_h, feat_w = shape
        feats.append(
            torch.randn(batch_size * num_cams, channels, feat_h, feat_w,
                        dtype=torch.float32, device=device))

    lidar2img = torch.eye(
        4, dtype=torch.float32, device=device).view(1, 1, 4, 4).repeat(
            batch_size, num_cams, 1, 1)
    can_bus = torch.zeros(
        batch_size, 18, dtype=torch.float32, device=device)
    img_shape = torch.tensor(
        [img_h, img_w], dtype=torch.float32, device=device).view(
            1, 1, 2).repeat(batch_size, num_cams, 1)
    return (*feats, lidar2img, can_bus, img_shape)

