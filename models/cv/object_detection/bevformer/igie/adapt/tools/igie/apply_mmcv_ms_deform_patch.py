#!/usr/bin/env python3
"""Apply mmcv ms_deform export patch (requires write access to mmcv/)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT = REPO_ROOT / 'mmcv/mmcv/ops/multi_scale_deform_attn.py'


def patch_file(path: Path) -> None:
    text = path.read_text()

    old_assert = "        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value\n"
    new_assert = (
        "        if not torch.compiler.is_compiling() and not torch.onnx.is_in_onnx_export():\n"
        "            assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value\n"
    )
    if old_assert in text and new_assert not in text:
        text = text.replace(old_assert, new_assert, 1)
        print('patched assert')

    old_branch = """        if torch.cuda.is_available() and value.is_cuda:
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)"""

    new_branch = """        if torch.cuda.is_available() and value.is_cuda \\
                and not torch.compiler.is_compiling() \\
                and not torch.onnx.is_in_onnx_export():
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)"""

    if old_branch in text:
        text = text.replace(old_branch, new_branch, 1)
        print('patched cuda branch')
    elif 'is_compiling' in text:
        print('skip (already patched)')
    else:
        raise SystemExit('unexpected file content; patch manually')

    path.write_text(text)
    print(f'ok: {path}')


def patch_pytorch_single_level(path: Path) -> None:
    """Avoid split_with_sizes in single-level ms_deform export path."""
    text = path.read_text()
    marker = '# Single-level path avoids split_with_sizes'
    if marker in text:
        print('skip pytorch single-level (already patched)')
        return

    old_body = """    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ =\\
        sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes],
                             dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        # bs, H_*W_, num_heads, embed_dims ->
        # bs, H_*W_, num_heads*embed_dims ->
        # bs, num_heads*embed_dims, H_*W_ ->
        # bs*num_heads, embed_dims, H_, W_
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(
            bs * num_heads, embed_dims, H_, W_)
        # bs, num_queries, num_heads, num_points, 2 ->
        # bs, num_heads, num_queries, num_points, 2 ->
        # bs*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :,
                                          level].transpose(1, 2).flatten(0, 1)
        # bs*num_heads, embed_dims, num_queries, num_points
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (bs, num_queries, num_heads, num_levels, num_points) ->
    # (bs, num_heads, num_queries, num_levels, num_points) ->
    # (bs, num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) *
              attention_weights).sum(-1).view(bs, num_heads * embed_dims,
                                              num_queries)
    return output.transpose(1, 2).contiguous()"""

    new_body = """    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ =\\
        sampling_locations.shape
    sampling_grids = 2 * sampling_locations - 1

    def _sample_one_level(value_l_, level, H_, W_):
        value_l_ = value_l_.flatten(2).transpose(1, 2).reshape(
            bs * num_heads, embed_dims, H_, W_)
        sampling_grid_l_ = sampling_grids[:, :, :,
                                          level].transpose(1, 2).flatten(0, 1)
        return F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)

    # Single-level path avoids split_with_sizes (igie / torch.export).
    if value_spatial_shapes.shape[0] == 1:
        H_ = int(value_spatial_shapes[0, 0])
        W_ = int(value_spatial_shapes[0, 1])
        sampling_value_list = [_sample_one_level(value, 0, H_, W_)]
    else:
        value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes],
                                 dim=1)
        sampling_value_list = []
        for level, (H_, W_) in enumerate(value_spatial_shapes):
            sampling_value_list.append(
                _sample_one_level(value_list[level], level, int(H_), int(W_)))

    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) *
              attention_weights).sum(-1).view(bs, num_heads * embed_dims,
                                              num_queries)
    return output.transpose(1, 2).contiguous()"""

    if old_body not in text:
        print('skip pytorch single-level (body already customized)')
        return

    path.write_text(text.replace(old_body, new_body, 1))
    print('patched pytorch single-level path')


def main():
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    patch_file(target)
    patch_pytorch_single_level(target)


if __name__ == '__main__':
    main()
