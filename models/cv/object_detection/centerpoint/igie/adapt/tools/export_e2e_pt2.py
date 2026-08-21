#!/usr/bin/env python3
"""Export CenterPoint PointPillars e2e (PFN+Scatter+RPN+Head) to torch.export *.pt2.

Decoration stays in PyTorch. Inputs:
  features [N, 20, 10]  (already decorated)
  coors    [N, 4] int32
Outputs: 36 flat head tensors (6 tasks x reg/height/dim/rot/vel/hm).

Batch is fixed to 1; num_pillars (N) is dynamic.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from torch import nn
from torch.export.dynamic_shapes import Dim

from det3d.models import build_detector
from det3d.torchie import Config
from det3d.torchie.trainer import load_checkpoint

HEAD_KEYS = ("reg", "height", "dim", "rot", "vel", "hm")


class ScatterBatch1(nn.Module):
    """Export-friendly BEV scatter for batch=1 (index_copy, no Python loop)."""

    def __init__(self, nx: int = 512, ny: int = 512, nchannels: int = 64):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.nchannels = nchannels

    def forward(self, voxel_features: torch.Tensor, coors: torch.Tensor) -> torch.Tensor:
        canvas = torch.zeros(
            self.nchannels,
            self.nx * self.ny,
            dtype=voxel_features.dtype,
            device=voxel_features.device,
        )
        indices = coors[:, 2].to(torch.int64) * self.nx + coors[:, 3].to(torch.int64)
        canvas = canvas.index_copy(1, indices, voxel_features.t())
        return canvas.view(1, self.nchannels, self.ny, self.nx)


class PointPillarsE2E(nn.Module):
    """PFN + Scatter(batch=1) + Neck + CenterHead → flat 36 tensors."""

    def __init__(self, model, nx: int = 512, ny: int = 512):
        super().__init__()
        self.pfn_layers = model.reader.pfn_layers
        self.scatter = ScatterBatch1(nx=nx, ny=ny, nchannels=64)
        self.neck = model.neck
        self.head = model.bbox_head

    def forward(self, features: torch.Tensor, coors: torch.Tensor):
        x = features
        for pfn in self.pfn_layers:
            x = pfn(x)
        x = x.squeeze(1)
        bev = self.scatter(x, coors)
        x = self.neck(bev)
        preds = self.head(x)
        outs = []
        for pred in preds:
            for key in HEAD_KEYS:
                outs.append(pred[key])
        return tuple(outs)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        default="configs/nusc/pp/nusc_centerpoint_pp_02voxel_two_pfn_10sweep_demo_mini.py",
    )
    p.add_argument("--checkpoint", default="./latest.pth")
    p.add_argument("--pt2-path", default="./torch_model/pp_e2e.pt2")
    p.add_argument("--example-n", type=int, default=4096, help="example pillar count")
    p.add_argument("--nx", type=int, default=512)
    p.add_argument("--ny", type=int, default=512)
    p.add_argument("--min-pillars", type=int, default=1)
    p.add_argument("--max-pillars", type=int, default=60000)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.eval().cuda()

    wrapper = PointPillarsE2E(model, nx=args.nx, ny=args.ny).eval()
    n = args.example_n
    features = torch.zeros(n, 20, 10, dtype=torch.float32, device="cuda")
    coors = torch.zeros(n, 4, dtype=torch.int32, device="cuda")
    # Valid unique scatter indices for example (avoid collisions).
    for i in range(n):
        coors[i, 2] = i // args.nx
        coors[i, 3] = i % args.nx

    num_pillars = Dim("num_pillars", min=args.min_pillars, max=args.max_pillars)
    dynamic_shapes = {
        "features": {0: num_pillars},
        "coors": {0: num_pillars},
    }

    out = Path(args.pt2_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        ep = torch.export.export(
            wrapper,
            (features, coors),
            dynamic_shapes=dynamic_shapes,
        )
        torch.export.save(ep, str(out))

    print(f"saved {out} (dynamic num_pillars [{args.min_pillars}, {args.max_pillars}])")


if __name__ == "__main__":
    main()
