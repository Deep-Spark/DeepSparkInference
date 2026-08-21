#!/usr/bin/env python3
"""Revert mmcv DCNv2 torchvision file patch (optional cleanup)."""

from __future__ import annotations

import sys
from pathlib import Path


def _resolve_mmcv_dcn_path() -> Path:
    try:
        import mmcv.ops.modulated_deform_conv as mdc  # noqa: WPS433
    except ImportError as exc:
        raise SystemExit('mmcv not importable; activate BEVFormer env first') from exc
    return Path(mdc.__file__)


def revert_torchvision_patch(path: Path) -> None:
    text = path.read_text()
    if 'torchvision.ops import deform_conv2d' not in text:
        print(f'skip (no torchvision patch): {path}')
        return

    patched_forward = """    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        out = self.conv_offset(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        if torch.compiler.is_compiling() or torch.onnx.is_in_onnx_export():
            from torchvision.ops import deform_conv2d
            x = x.type_as(offset)
            weight = self.weight.type_as(x)
            mask = mask.type_as(x)
            bias = self.bias.type_as(x) if self.bias is not None else None
            return deform_conv2d(
                x,
                offset,
                weight,
                bias=bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                mask=mask,
            )
        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)"""

    original_forward = """    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        out = self.conv_offset(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)"""

    if patched_forward not in text:
        raise SystemExit(f'torchvision patch present but unexpected format: {path}')

    path.write_text(text.replace(patched_forward, original_forward, 1))
    print(f'reverted torchvision patch: {path}')


def main():
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else _resolve_mmcv_dcn_path()
    revert_torchvision_patch(target)


if __name__ == '__main__':
    main()
