"""Pure-torch modulated deformable conv for torch.export / IGIE."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _pair(x):
    if isinstance(x, int):
        return x, x
    return x


def modulated_deform_conv2d_pytorch(
    input: torch.Tensor,
    offset: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride=1,
    padding=0,
    dilation=1,
    groups: int = 1,
    deform_groups: int = 1,
) -> torch.Tensor:
    """DCNv2 reference path using grid_sample (export / IGIE friendly)."""
    return modulated_deform_conv2d_n_kernels(
        input,
        offset,
        mask,
        weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        deform_groups=deform_groups,
        num_kernels=None,
    )


def modulated_deform_conv2d_n_kernels(
    input: torch.Tensor,
    offset: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride=1,
    padding=0,
    dilation=1,
    groups: int = 1,
    deform_groups: int = 1,
    num_kernels: int | None = None,
) -> torch.Tensor:
    """First num_kernels DCN taps with accumulation (None = full 3x3 kernel)."""
    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    dil_h, dil_w = _pair(dilation)

    n, in_ch, in_h, in_w = input.shape
    out_ch, w_in_ch, kH, kW = weight.shape
    out_h = (in_h + 2 * pad_h - (dil_h * (kH - 1) + 1)) // stride_h + 1
    out_w = (in_w + 2 * pad_w - (dil_w * (kW - 1) + 1)) // stride_w + 1

    input_g = input.view(n, groups, in_ch // groups, in_h, in_w)
    weight_g = weight.view(groups, out_ch // groups, w_in_ch, kH, kW)
    out_ch_per_g = out_ch // groups

    base_y = torch.arange(out_h, device=input.device, dtype=input.dtype).view(1, 1, out_h, 1)
    base_x = torch.arange(out_w, device=input.device, dtype=input.dtype).view(1, 1, 1, out_w)

    total_kernels = kH * kW
    if num_kernels is None:
        num_kernels = total_kernels
    else:
        num_kernels = min(int(num_kernels), total_kernels)

    group_outs = []
    for g in range(groups):
        # Full-tensor add per group; avoid out[:, g] = out[:, g] + ... (IGIE lowering bug).
        g_out = input.new_zeros(n, out_ch_per_g, out_h, out_w)
        inp = input_g[:, g]
        for ki in range(num_kernels):
            kh, kw = divmod(ki, kW)
            dg = g * deform_groups // groups
            o_base = dg * kH * kW + ki
            off_y = offset[:, 2 * o_base:2 * o_base + 1]
            off_x = offset[:, 2 * o_base + 1:2 * o_base + 2]
            m = mask[:, o_base:o_base + 1]

            iy = base_y * stride_h + kh * dil_h + off_y - pad_h
            ix = base_x * stride_w + kw * dil_w + off_x - pad_w
            gx = 2.0 * (ix.squeeze(1) + 0.5) / in_w - 1.0
            gy = 2.0 * (iy.squeeze(1) + 0.5) / in_h - 1.0
            grid = torch.stack([gx, gy], dim=-1)

            sampled = F.grid_sample(
                inp,
                grid,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False,
            )
            w = weight_g[g, :, :, kh, kw]
            s = (sampled * m).flatten(2)
            g_out = g_out + torch.matmul(w, s).reshape(n, out_ch_per_g, out_h, out_w)
        group_outs.append(g_out)

    out = torch.cat(group_outs, dim=1)
    if bias is not None:
        out = out + bias.view(1, -1, 1, 1)
    return out


def modulated_deform_conv2d_one_kernel(
    input: torch.Tensor,
    offset: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
    stride=1,
    padding=0,
    dilation=1,
    groups: int = 1,
    deform_groups: int = 1,
    kernel_idx: int = 0,
    group_idx: int = 0,
) -> torch.Tensor:
    """Single DCN kernel: grid_sample -> mask -> matmul(weight[:,:,kh,kw], s)."""
    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    dil_h, dil_w = _pair(dilation)

    n, in_ch, in_h, in_w = input.shape
    out_ch, w_in_ch, kH, kW = weight.shape
    out_h = (in_h + 2 * pad_h - (dil_h * (kH - 1) + 1)) // stride_h + 1
    out_w = (in_w + 2 * pad_w - (dil_w * (kW - 1) + 1)) // stride_w + 1

    input_g = input.view(n, groups, in_ch // groups, in_h, in_w)
    weight_g = weight.view(groups, out_ch // groups, w_in_ch, kH, kW)

    base_y = torch.arange(out_h, device=input.device, dtype=input.dtype).view(1, 1, out_h, 1)
    base_x = torch.arange(out_w, device=input.device, dtype=input.dtype).view(1, 1, 1, out_w)

    g = group_idx
    inp = input_g[:, g]
    kh, kw = divmod(kernel_idx, kW)
    dg = g * deform_groups // groups
    o_base = dg * kH * kW + kernel_idx
    off_y = offset[:, 2 * o_base:2 * o_base + 1]
    off_x = offset[:, 2 * o_base + 1:2 * o_base + 2]
    m = mask[:, o_base:o_base + 1]

    iy = base_y * stride_h + kh * dil_h + off_y - pad_h
    ix = base_x * stride_w + kw * dil_w + off_x - pad_w
    gx = 2.0 * (ix.squeeze(1) + 0.5) / in_w - 1.0
    gy = 2.0 * (iy.squeeze(1) + 0.5) / in_h - 1.0
    grid = torch.stack([gx, gy], dim=-1)

    sampled = F.grid_sample(
        inp,
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False,
    )
    w = weight_g[g, :, :, kh, kw]
    s = (sampled * m).flatten(2)
    return torch.matmul(w, s).reshape(n, w.shape[0], out_h, out_w)


def _in_export_trace() -> bool:
    """True during torch.export / dynamo tracing (not plain eager eval)."""
    if torch.onnx.is_in_onnx_export():
        return True
    if torch.compiler.is_compiling():
        return True
    try:
        import torch._dynamo as dynamo
        if dynamo.is_compiling():
            return True
    except Exception:
        pass
    return False


def install_dcn_export_fallback() -> None:
    """Monkeypatch mmcv DCNv2 to use grid_sample path during export."""
    import mmcv.ops.modulated_deform_conv as mdc

    if getattr(mdc.ModulatedDeformConv2dPack, '_igie_export_patched', False):
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_offset(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        if _in_export_trace():
            return modulated_deform_conv2d_pytorch(
                x,
                offset,
                mask,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
                self.deform_groups,
            )
        return mdc.modulated_deform_conv2d(
            x,
            offset,
            mask,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.deform_groups,
        )

    mdc.ModulatedDeformConv2dPack.forward = forward
    mdc.ModulatedDeformConv2dPack._igie_export_patched = True
    print('installed DCNv2 grid_sample export fallback')
