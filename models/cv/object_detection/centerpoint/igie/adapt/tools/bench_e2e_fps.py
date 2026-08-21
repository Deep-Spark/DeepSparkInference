#!/usr/bin/env python3
"""Bench CenterPoint e2e single-SO IGIE latency / FPS."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from det3d.datasets import build_dataloader, build_dataset  # noqa: E402
from det3d.models import build_detector  # noqa: E402
from det3d.torchie import Config  # noqa: E402
from det3d.torchie.apis import example_to_device  # noqa: E402
from det3d.torchie.trainer import load_checkpoint  # noqa: E402
from det3d.models.detectors.point_pillars import torch_cuda_to_igie  # noqa: E402


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _ms(fn, warmup: int, runs: int) -> float:
    for _ in range(warmup):
        fn()
    _sync()
    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    _sync()
    return (time.perf_counter() - t0) * 1000.0 / max(runs, 1)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--igie-config",
        default="configs/nusc/pp/nusc_centerpoint_pp_02voxel_two_pfn_10sweep_demo_mini_igie_e2e.py",
    )
    p.add_argument("--checkpoint", default="latest.pth")
    p.add_argument("--e2e-so", default="pp_e2e_fp16_ixinfer.so")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--sample-idx", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.igie_config)
    dataset = build_dataset(cfg.data.val)
    loader = build_dataloader(
        dataset, batch_size=1, workers_per_gpu=0, dist=False, shuffle=False
    )
    example = None
    for i, batch in enumerate(loader):
        if i == args.sample_idx:
            example = example_to_device(
                batch, device=torch.device("cuda"), non_blocking=False
            )
            break
    if example is None:
        raise SystemExit(f"sample_idx={args.sample_idx} out of range")

    n_pillars = int(example["voxels"].shape[0])
    print(f"bench sample_idx={args.sample_idx} num_pillars={n_pillars}")

    cfg.test_cfg["e2e_engine_path"] = args.e2e_so
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.eval().cuda()

    data = dict(
        features=example["voxels"],
        num_voxels=example["num_points"],
        coors=example["coordinates"],
        batch_size=1,
        input_shape=example["shape"][0],
    )

    def run_wall():
        with torch.no_grad():
            preds = model._run_e2e(data)
            for pred in preds:
                for v in pred.values():
                    if torch.is_tensor(v) and v.is_cuda:
                        _ = v.reshape(-1)[0].item()
                        break
                break

    decorated = model.reader.decorate_features(
        data["features"], data["num_voxels"], data["coors"]
    )
    coors = data["coors"].to(torch.int32).contiguous()
    feats_nd = torch_cuda_to_igie(decorated, model.device)
    coors_nd = torch_cuda_to_igie(coors, model.device)

    def run_engine():
        outs = model.e2e_vm["main"](feats_nd, coors_nd)
        _ = outs[0].numpy().ravel()[0]

    wall = _ms(run_wall, args.warmup, args.runs)
    eng = _ms(run_engine, args.warmup, args.runs)
    print("\n=== IGIE e2e bench ===")
    print(f"IGIE wall:        {wall:7.2f} ms  ({1000 / wall:5.1f} fps)")
    print(f"IGIE engine-only: {eng:7.2f} ms  ({1000 / eng:5.1f} fps)")


if __name__ == "__main__":
    main()
