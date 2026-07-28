"""Export-friendly SpatialCrossAttention (dense, fixed num_query)."""

from __future__ import annotations

import torch

from dcn_export_core import _in_export_trace


def spatial_cross_attention_export_forward(
    self,
    query,
    key=None,
    value=None,
    residual=None,
    query_pos=None,
    key_padding_mask=None,
    reference_points=None,
    spatial_shapes=None,
    reference_points_cam=None,
    bev_mask=None,
    level_start_index=None,
    **kwargs,
):
    """Dense SCA path: no nonzero / dynamic max_len (torch.export friendly)."""
    if key is None:
        key = query
    if value is None:
        value = key

    if residual is None:
        inp_residual = query
    if query_pos is not None:
        query = query + query_pos

    bs, num_query, _ = query.size()
    max_len = num_query

    # reference_points_cam: [num_cams, bs, num_query, D, 2]
    queries_rebatch = query.unsqueeze(1).expand(
        bs, self.num_cams, num_query, self.embed_dims).contiguous()
    reference_points_rebatch = reference_points_cam.permute(
        1, 0, 2, 3, 4).contiguous()

    num_cams, l, bs_key, embed_dims = key.shape
    assert num_cams == self.num_cams and bs_key == bs

    key = key.permute(2, 0, 1, 3).reshape(bs * self.num_cams, l, embed_dims)
    value = value.permute(2, 0, 1, 3).reshape(bs * self.num_cams, l, embed_dims)

    queries = self.deformable_attention(
        query=queries_rebatch.reshape(bs * self.num_cams, max_len, self.embed_dims),
        key=key,
        value=value,
        reference_points=reference_points_rebatch.reshape(
            bs * self.num_cams, max_len, reference_points_rebatch.size(3), 2),
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
    ).view(bs, self.num_cams, max_len, self.embed_dims)

    # Float-only mask aggregation (avoid bool any/>0/amax for IGIE).
    # bev_mask: [num_cams, bs, num_query, D] — 0/1 float after point_sampling patch
    cam_valid = bev_mask.to(dtype=queries.dtype).sum(dim=-1).clamp(min=0.0, max=1.0)
    valid = cam_valid.permute(1, 0, 2)  # [bs, num_cams, num_query]
    slots = (queries * valid.unsqueeze(-1)).sum(dim=1)
    count = cam_valid.permute(1, 2, 0).sum(-1).clamp(min=1.0)
    slots = slots / count.unsqueeze(-1)
    slots = self.output_proj(slots)

    return self.dropout(slots) + inp_residual


def install_sca_export_patch() -> None:
    """Use dense SCA during torch.export (avoid GuardOnDataDependentSymNode)."""
    from projects.mmdet3d_plugin.bevformer.modules.spatial_cross_attention import (
        SpatialCrossAttention,
    )

    if getattr(SpatialCrossAttention, '_igie_export_patched', False):
        return

    _orig_sca_forward = SpatialCrossAttention.forward

    def _sca_forward(self, *args, **kwargs):
        if _in_export_trace():
            return spatial_cross_attention_export_forward(self, *args, **kwargs)
        return _orig_sca_forward(self, *args, **kwargs)

    SpatialCrossAttention.forward = _sca_forward
    SpatialCrossAttention._igie_export_patched = True
    print('installed SpatialCrossAttention dense export patch')
