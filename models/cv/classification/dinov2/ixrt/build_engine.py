#!/usr/bin/env python3
"""把 onnx-community/dinov2-base 的动态图固化为静态 shape 并构建 ixRT engine。

原始 ONNX 的输入是 pixel_values[batch_size, num_channels, height, width]，
DINOv2 的位置编码按 518x518（37x37 patch）训练，前向里用 bicubic Resize 插值到实际
分辨率。固化输入尺寸后这个 Resize 会被常量折叠成一个固定的位置编码张量，
ixRT 就不需要支持动态 cubic Resize 了 —— 这一步是构建成功的前提。
"""

import argparse
import collections
import os
import sys
import tempfile

import onnx

from ixrt_helper import import_trt, load_ixrt_plugin, make_logger


def load_model(path):
    """加载 ONNX 并把外置权重（.onnx.data）一并读进内存。

    torch.onnx.export 的新版实现会把权重拆到同名 .onnx.data 里，只搬主文件会导致
    后续读取报 'should be stored in ....onnx.data, but it is not regular file'。
    """
    model = onnx.load(path, load_external_data=True)
    sidecar = path + ".data"
    if os.path.exists(sidecar):
        size_mb = os.path.getsize(sidecar) / 1024 / 1024
        print(f"[info] 检测到外置权重 {os.path.basename(sidecar)} ({size_mb:.1f} MB)，已内联加载")
    return model


def save_model(model, path):
    """保存为自包含单文件；超过 protobuf 2GB 上限时才退回外置权重。"""
    try:
        onnx.save(model, path, save_as_external_data=False)
        return
    except Exception as exc:  # noqa: BLE001 - 只关心是否触达 2GB 上限
        print(f"[warn] 单文件保存失败({exc})，改用外置权重")
    location = os.path.basename(path) + ".data"
    sidecar = os.path.join(os.path.dirname(path) or ".", location)
    if os.path.exists(sidecar):
        os.remove(sidecar)
    onnx.save(model, path, save_as_external_data=True, all_tensors_to_one_file=True, location=location)


def describe(model, tag):
    print(f"--- {tag} ---")
    for kind, tensors in (("input ", model.graph.input), ("output", model.graph.output)):
        for tensor in tensors:
            dims = [d.dim_param or d.dim_value for d in tensor.type.tensor_type.shape.dim]
            dtype = onnx.TensorProto.DataType.Name(tensor.type.tensor_type.elem_type)
            print(f"  {kind}: {tensor.name} {dims} {dtype}")
    ops = collections.Counter(node.op_type for node in model.graph.node)
    print(f"  节点总数: {len(model.graph.node)}")
    print(f"  算子分布: {dict(sorted(ops.items(), key=lambda kv: -kv[1]))}")
    risky = {op: ops[op] for op in ("Resize", "If", "Loop", "NonZero", "ScatterND") if op in ops}
    if risky:
        print(f"  [warn] 仍存在动态/不易支持的算子: {risky}")
    custom = collections.Counter(
        f"{node.domain}::{node.op_type}" for node in model.graph.node if node.domain not in ("", "ai.onnx")
    )
    if custom:
        print(f"  [warn] 存在非标准域算子，ixRT 无对应 plugin，解析会失败: {dict(custom)}")
    return ops


def freeze_input(model, name, dims):
    target = next((t for t in model.graph.input if t.name == name), None)
    if target is None:
        available = [t.name for t in model.graph.input]
        raise SystemExit(f"ONNX 里没有输入 '{name}'，实际输入为 {available}")
    shape = target.type.tensor_type.shape.dim
    if len(shape) != len(dims):
        raise SystemExit(f"输入 '{name}' 是 {len(shape)} 维，但给了 {len(dims)} 个尺寸")
    for dim, value in zip(shape, dims):
        dim.ClearField("dim_param")
        dim.dim_value = int(value)
    # 输出维度置为未知交给下游重新推导，避免残留符号维度。
    # 注意保留 shape 字段本身：整个 ClearField('shape') 会让 onnx checker 报
    # "Field 'shape' of 'type' is required but missing"，onnxsim 会直接失败。
    for tensor in model.graph.output:
        for dim in tensor.type.tensor_type.shape.dim:
            dim.ClearField("dim_param")
            dim.ClearField("dim_value")
    del model.graph.value_info[:]
    return model


def simplify_with_onnxsim(model, name, dims):
    try:
        from onnxsim import simplify
    except ImportError:
        return None
    for kwargs in ({"overwrite_input_shapes": {name: list(dims)}}, {"input_shapes": {name: list(dims)}}, {}):
        try:
            simplified, ok = simplify(model, **kwargs)
        except TypeError:
            continue
        except Exception as exc:  # noqa: BLE001 - onnxsim 失败就回退到 ORT
            print(f"[warn] onnxsim 失败({exc})，回退 onnxruntime 折叠")
            return None
        if not ok:
            print("[warn] onnxsim 自检未通过，但仍使用简化结果（后续会做精度比对）")
        return simplified
    print("[warn] onnxsim 参数不兼容，回退 onnxruntime 折叠")
    return None


def simplify_with_ort(model, work_dir):
    try:
        import onnxruntime as ort
    except ImportError:
        raise SystemExit("需要 onnxsim 或 onnxruntime 之一来做常量折叠：pip3 install onnxsim onnxruntime")
    raw_path = os.path.join(work_dir, "_frozen_raw.onnx")
    opt_path = os.path.join(work_dir, "_frozen_opt.onnx")
    onnx.save(model, raw_path, save_as_external_data=False)
    options = ort.SessionOptions()
    # 只能用 BASIC：常量折叠属于 Level 1，足以折掉位置编码的 Resize；
    # EXTENDED(Level 2) 会额外做 MatMulScaleFusion / SkipLayerNormFusion 等融合，
    # 生成 com.microsoft 私有域算子（FusedMatMul、SkipLayerNormalization），
    # ixRT 没有对应 plugin，解析阶段会直接报 "Plugin not found"。
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    options.optimized_model_filepath = opt_path
    ort.InferenceSession(raw_path, options, providers=["CPUExecutionProvider"])
    return onnx.load(opt_path)


def build_engine(onnx_path, engine_path, precision, verbose):
    trt = import_trt()
    logger = make_logger(trt, verbose)
    load_ixrt_plugin(trt, logger)

    builder = trt.Builder(logger)
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(onnx_path):
        for i in range(getattr(parser, "num_errors", 0)):
            print(f"[parser] {parser.get_error(i)}")
        raise SystemExit(f"解析 ONNX 失败: {onnx_path}")

    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)

    print(f"[build] precision={precision} layers={network.num_layers} 开始构建 ...")
    plan = builder.build_serialized_network(network, config)
    if not plan:
        raise SystemExit("构建 engine 失败，加 --verbose 看详细日志")
    with open(engine_path, "wb") as f:
        f.write(plan)
    size_mb = os.path.getsize(engine_path) / 1024 / 1024
    print(f"[ok] engine 已保存: {engine_path} ({size_mb:.1f} MB)")


def parse_args():
    parser = argparse.ArgumentParser(description="DINOv2 ONNX -> ixRT engine")
    parser.add_argument("--onnx", required=True, help="原始 model.onnx 路径")
    parser.add_argument("--out-dir", default="./checkpoints")
    parser.add_argument("--input-name", default="pixel_values")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=224, help="必须是 14 的倍数（patch_size=14）")
    parser.add_argument("--precision", choices=["fp16", "fp32", "int8"], default="fp16")
    parser.add_argument("--skip-simplify", action="store_true", help="已有静态 ONNX 时跳过固化")
    parser.add_argument("--only-onnx", action="store_true", help="只导出静态 ONNX，不构建 engine")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.imgsz % 14 != 0:
        raise SystemExit(f"--imgsz 必须是 14 的倍数，当前 {args.imgsz}")
    os.makedirs(args.out_dir, exist_ok=True)

    dims = [args.batch, 3, args.imgsz, args.imgsz]
    tag = f"bs{args.batch}_{args.imgsz}"
    static_onnx = os.path.join(args.out_dir, f"dinov2_base_{tag}.onnx")
    engine_path = os.path.join(args.out_dir, f"dinov2_base_{tag}_{args.precision}.engine")

    if args.skip_simplify:
        model = load_model(args.onnx)
        ops = describe(model, "静态 ONNX（跳过固化）")
        dynamic = [
            t.name
            for t in model.graph.input
            if any(d.dim_param or d.dim_value == 0 for d in t.type.tensor_type.shape.dim)
        ]
        if dynamic:
            raise SystemExit(f"输入 {dynamic} 仍是动态 shape，不能用 --skip-simplify，去掉这个参数重跑")
        if ops.get("Resize"):
            print("[warn] 图里仍有 Resize，ixRT 构建可能失败")
        if os.path.abspath(args.onnx) != os.path.abspath(static_onnx):
            save_model(model, static_onnx)
            print(f"[ok] 已转存为自包含单文件: {static_onnx}")
        else:
            static_onnx = args.onnx
    else:
        model = load_model(args.onnx)
        describe(model, "原始 ONNX")
        model = freeze_input(model, args.input_name, dims)
        print(f"[step] 固化输入 {args.input_name} -> {dims}，常量折叠位置编码 Resize ...")
        simplified = simplify_with_onnxsim(model, args.input_name, dims)
        if simplified is None:
            with tempfile.TemporaryDirectory() as work_dir:
                simplified = simplify_with_ort(model, work_dir)
        save_model(simplified, static_onnx)
        ops = describe(simplified, "静态 ONNX")
        if ops.get("Resize"):
            print("[warn] Resize 未被折叠，engine 构建可能失败；确认 onnxruntime 已安装后重跑")
        print(f"[ok] 静态 ONNX 已保存: {static_onnx}")

    if args.only_onnx:
        return
    build_engine(static_onnx, engine_path, args.precision, args.verbose)


if __name__ == "__main__":
    sys.exit(main())
