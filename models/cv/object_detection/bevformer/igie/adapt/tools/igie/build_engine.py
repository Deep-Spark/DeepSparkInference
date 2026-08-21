# Copyright (c) 2024, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# Copied from deformable-detr/build_engine.py for BEVFormer IGIE deployment.

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program


def apply_igie_gemm_rewriter_patch():
    """Wrap legalize_float16 rewrite_call to skip bad GEMM fusions (e.g. batched bmm)."""
    from tvm.relax.transform.iluvatar import legalize_float16 as lf16

    if getattr(lf16, '_gemm_rewrite_call_patched', False):
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


def _prepare_relax_mod(relax_mod):
    """Decompose inference BN before iluvatar pipeline; sanity-check IR."""
    text = relax_mod.script()
    bn_before = text.count('batch_norm')
    relax_mod = relax.transform.DecomposeOpsForInference()(relax_mod)
    text = relax_mod.script()
    bn_after = text.count('batch_norm')
    bad_bn = text.count('batch_norm(metadata')
    print(f'batch_norm in Relax IR: {bn_before} -> {bn_after} after DecomposeOpsForInference')
    if bad_bn:
        print(
            f'WARNING: found {bad_bn} batch_norm(data=constant) nodes; '
            'DCN subgraph likely constant-folded. Re-export backbone.pt2 with '
            'install_dcn_export_fallback() + install_frozen_bn_export().')
    return relax_mod


def _mod_has_conv2d(relax_mod) -> bool:
    text = relax_mod.script()
    return 'nn.conv2d' in text or 'nn.conv2d_transpose' in text


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='ExportedProgram (.pt2) path')
    parser.add_argument('--engine_path', type=str, required=True, help='Output .so path')
    parser.add_argument(
        '--precision',
        type=str,
        choices=['fp32', 'fp16', 'int8'],
        default='fp32',
        help='Inference precision',
    )
    parser.add_argument(
        '--layout',
        type=str,
        choices=['NHWC', 'NCHW'],
        default='NHWC',
        help='conv layout for iluvatar pipeline (ixinfer qdconv requires NHWC)',
    )
    parser.add_argument(
        '--debug-dump',
        type=str,
        default=None,
        help='dump Relax IR after each pipeline phase to this directory',
    )
    parser.add_argument(
        '--target-options',
        type=str,
        default=None,
        help='iluvatar target options; fp16 ixinfer fusion needs "-libs=cudnn,cublas,ixinfer"',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='rebuild even if engine_path already exists',
    )
    parser.add_argument(
        '--no-deform-fusion',
        action='store_true',
        help='skip DCN/MSDeform subgraph fusion (debug / fallback builds)',
    )
    return parser.parse_args()


def _engine_stale(model_path: str, engine_path: str) -> bool:
    if not os.path.isfile(engine_path):
        return True
    return os.path.getmtime(model_path) > os.path.getmtime(engine_path)


def main():
    args = parse_args()

    if os.path.exists(args.engine_path):
        if args.force or _engine_stale(args.model_path, args.engine_path):
            if not args.force:
                print(
                    f'{args.engine_path} is older than {args.model_path}; rebuilding')
            os.remove(args.engine_path)
        else:
            print(f'engine already exists: {args.engine_path}')
            return

    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(f'model not found: {args.model_path}')

    if not args.model_path.endswith('.pt2'):
        raise ValueError(f'expected .pt2, got: {args.model_path}')

    target_options = args.target_options

    print(f'loading ExportedProgram from {args.model_path}')
    exported_program = torch.export.load(args.model_path)

    print('importing ExportedProgram via Relax torch frontend')
    relax_mod = from_exported_program(exported_program, keep_params_as_input=False)
    relax_mod = _prepare_relax_mod(relax_mod)

    # Fuse exported grid_sample subgraphs before layout conversion (pipeline repeats
    # the same pass when grid_sample remains — e.g. direct relax.build without this).
    from tvm.relax.transform.iluvatar.deform_conv_rewriters import (
        MSDeformAttnRewriter,
        ModulatedDeformConvRewriter,
    )
    from tvm.relax.transform.iluvatar.deform_conv_pipeline import (
        mod_has_deform_grid_sample_export,
    )

    n_gs_before = relax_mod.script().count('grid_sample') + relax_mod.script().count(
        'image.grid_sample')
    skip_pipeline_deform_fusion = args.no_deform_fusion
    if args.no_deform_fusion:
        print(f'deform fusion: skipped (--no-deform-fusion; grid_sample~{n_gs_before})')
    elif mod_has_deform_grid_sample_export(relax_mod):
        relax_mod = ModulatedDeformConvRewriter()(relax_mod)
        relax_mod = MSDeformAttnRewriter()(relax_mod)
        text_after = relax_mod.script()
        print(
            f'deform fusion: grid_sample~{n_gs_before} -> '
            f"mdcn={text_after.count('modulated_deform_conv')} "
            f"ms_deform={text_after.count('ms_deform_attn')}"
        )
        skip_pipeline_deform_fusion = True
    else:
        print('deform fusion: skipped (no grid_sample in Relax IR)')

    if target_options is None and args.precision == 'fp16':
        if _mod_has_conv2d(relax_mod):
            target_options = '-libs=cudnn,cublas,ixinfer'
        else:
            target_options = '-libs=ixinfer'

    target = tvm.target.iluvatar(model='MR', options=target_options)
    print(f'iluvatar target: {target}')
    debug_dump = Path(args.debug_dump).resolve() if args.debug_dump else None
    if debug_dump is not None:
        debug_dump.mkdir(parents=True, exist_ok=True)
        print(f'debug IR dumps -> {debug_dump}')

    if args.precision == 'fp16':
        print('applying igie GEMM rewriter patch (skip batched matmul fusion)')
        apply_igie_gemm_rewriter_patch()

    if args.precision == 'fp16':
        print(
            f'fp16 pipeline: target.libs={target.libs}, layout={args.layout}')

    skip_convert_layout = not _mod_has_conv2d(relax_mod)
    if skip_convert_layout:
        print('skipping ConvertLayout (no conv2d in Relax IR)')

    pipeline = relax.get_pipeline(
        'iluvatar_flexible_build',
        target=target,
        layout=args.layout,
        precision=args.precision,
        use_cudnn='cudnn' in target.libs,
        use_cublas='cublas' in target.libs,
        use_ixinfer='ixinfer' in target.libs,
        debug_dump=debug_dump,
        skip_convert_layout=skip_convert_layout,
        skip_deform_fusion=skip_pipeline_deform_fusion,
    )

    print(
        f'building engine with precision={args.precision}, '
        f'layout={args.layout}, target={target}')
    ex = relax.build(
        relax_mod,
        target=target,
        pipeline=pipeline,
        layout=args.layout,
        precision=args.precision,
        verbose=debug_dump is not None,
    )

    print(f'exporting engine to {args.engine_path}')
    ex.export_library(args.engine_path)
    print('done.')


if __name__ == '__main__':
    main()
