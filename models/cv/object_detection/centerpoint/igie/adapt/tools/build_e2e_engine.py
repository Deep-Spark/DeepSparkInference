#!/usr/bin/env python3
"""Build CenterPoint e2e (PFN+Scatter+RPN+Head) IGIE engine from torch.export *.pt2."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program


def apply_igie_gemm_rewriter_patch():
    from tvm.relax.transform.iluvatar import legalize_float16 as lf16

    if getattr(lf16, "_gemm_rewrite_call_patched", False):
        return
    _orig_rewrite_call = lf16.rewrite_call

    def _safe_rewrite_call(pattern, rewriter, func):
        def _safe_rewriter(expr, matches):
            try:
                return rewriter(expr, matches)
            except AssertionError:
                return expr

        return _orig_rewrite_call(pattern, _safe_rewriter, func)

    lf16.rewrite_call = _safe_rewrite_call
    lf16._gemm_rewrite_call_patched = True


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pt2-path", default="./torch_model/pp_e2e.pt2")
    p.add_argument("--engine-path", default="./pp_e2e_fp16_ixinfer.so")
    p.add_argument("--network-name", default="e2e")
    p.add_argument("--precision", choices=["fp32", "fp16"], default="fp16")
    p.add_argument("--layout", choices=["NHWC", "NCHW"], default="NHWC")
    p.add_argument("--debug-dump", default="./igie_dump")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    engine_path = Path(args.engine_path).resolve()
    pt2_path = Path(args.pt2_path).resolve()

    if not pt2_path.is_file():
        raise FileNotFoundError(pt2_path)

    if engine_path.exists() and not args.force:
        if pt2_path.stat().st_mtime <= engine_path.stat().st_mtime:
            print(f"engine already exists: {engine_path}")
            return
        engine_path.unlink()

    dump_dir = Path(args.debug_dump).resolve() / args.network_name
    dump_dir.mkdir(parents=True, exist_ok=True)
    print(f"Relax IR dumps -> {dump_dir}")

    print(f"loading {pt2_path}")
    ep = torch.export.load(str(pt2_path))
    print("from_exported_program ...")
    relax_mod = from_exported_program(ep, keep_params_as_input=False)
    for p in relax_mod["main"].params:
        print(f"  input {p.name_hint}: {p.struct_info}")

    text = relax_mod.script()
    bn_before = text.count("batch_norm")
    relax_mod = relax.transform.DecomposeOpsForInference()(relax_mod)
    bn_after = relax_mod.script().count("batch_norm")
    print(f"batch_norm in Relax IR: {bn_before} -> {bn_after}")

    if args.precision == "fp16":
        apply_igie_gemm_rewriter_patch()

    target = tvm.target.iluvatar(
        model="MR", options="-libs=cudnn,cublas,ixinfer"
    )
    pipeline = relax.get_pipeline(
        "iluvatar_flexible_build",
        target=target,
        layout=args.layout,
        precision=args.precision,
        use_cudnn=True,
        use_cublas=True,
        use_ixinfer=True,
        debug_dump=dump_dir,
        skip_convert_layout=False,
    )

    print(
        f"building e2e engine: precision={args.precision}, layout={args.layout}, "
        f"target={target}"
    )
    ex = relax.build(
        relax_mod,
        target=target,
        pipeline=pipeline,
        layout=args.layout,
        precision=args.precision,
        verbose=True,
    )
    ex.export_library(str(engine_path))
    print(f"exported engine to {engine_path} ({engine_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
