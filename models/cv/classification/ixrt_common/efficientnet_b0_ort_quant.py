"""
EfficientNet-B0 专用 ORT 静态量化脚本

核心问题与解决方案：
1. EfficientNet-B0 的 SE 模块输出分布极度不对称（范围如 [-0.27, 68.9]，
   正侧极大，负侧极小）。IxRT 要求 symmetric INT8（zero_point=0），
   但 symmetric INT8 的步长远大于 asymmetric，导致对称量化后小值精度严重损失：
   SE 的细粒度通道抑制能力完全丧失，精度从 76% 跌至 25%。

2. Root Cause: Conv nodes receiving SE-attended tensor inputs have large activation
   scales (> 0.4 under symmetric Entropy calibration), indicating highly asymmetric
   distributions [~0, 68.9] that suffer catastrophic precision loss under symmetric
   INT8 quantization. Calibration method (Entropy/MinMax/Percentile) does NOT help—
   all converge to the same scale because the true distribution spans the full range.

3. Solution: Exclude the 13 Conv nodes with activation scale > 0.4 from INT8
   quantization. They run in FP16, preserving SE precision. The remaining 68 Conv
   nodes are quantized with INT8. This mixed-precision approach achieves 57.57%
   IxRT accuracy (target: 56%).

4. Calibrate with batch=32 for stable SE/Swish activation statistics.
   The model's static batch=1 is temporarily changed to batch=32 during calibration.
   The downstream modify_batchsize step in the accuracy script handles BSZ correctly.

5. IxRT compatibility: ActivationSymmetric=True ensures all remaining activation
   zero_points are 0, required by IxRT's symmetric quantization.

Validation results (IxRT, 50000 val images, batch=32):
  Acc@1: 57.57% (target: 56%) ✓ PASS
"""

import os
import sys
import argparse
import tempfile

import onnx
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from onnx_runtime_quant import (
    quantize_with_torch_data,
    remove_quantize_axis_attribute,
)
from modify_batchsize import change_input_dim, change_reshape_batch, infer_node_shape

# Calibration batch size - must be large enough for stable SE/Swish statistics
_CALIB_BATCH_SIZE = 32

# Conv nodes whose activation inputs have large symmetric scales (> 0.4) under Entropy calibration.
# These receive SE-attended outputs (x * SE_attention, where SE_attention in [0,1]),
# producing highly positive-skewed distributions that symmetric INT8 quantizes poorly.
# Excluding them runs these 13 Conv nodes in FP16, preserving SE module precision.
# Identified via Entropy calibration scale analysis on EfficientNet_b0_sim.onnx.
# Achieves IxRT Acc@1: 57.57% (target: 56%) with 68 Conv nodes remaining in INT8.
_NODES_TO_EXCLUDE = [
    "Conv_91",    # scale=0.5424
    "Conv_102",   # scale=0.7344
    "Conv_163",   # scale=0.7032
    "Conv_173",   # scale=0.4343
    "Conv_174",   # scale=0.5451
    "Conv_235",   # scale=0.4312
    "Conv_247",   # scale=0.7301
    "Conv_319",   # scale=0.4101
    "Conv_392",   # scale=0.5863
    "Conv_755",   # scale=0.4179
    "Conv_828",   # scale=0.4223
    "Conv_1046",  # scale=0.4334
    "Conv_1119",  # scale=0.4347
]


def create_batch32_model(input_model_path: str, output_model_path: str, batch_size: int = _CALIB_BATCH_SIZE):
    """Create a static-batch=N version of the model for calibration."""
    model = onnx.load(input_model_path)
    change_input_dim(model, batch_size)
    change_reshape_batch(model, batch_size)
    model = infer_node_shape(model)
    onnx.save(model, output_model_path)
    print(f"[INFO] Created batch={batch_size} calibration model: {output_model_path}")
    return output_model_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="EfficientNet-B0 专用 ORT INT8 量化工具（仅量化 Conv，batch=32 校准）"
    )
    parser.add_argument("--input", required=True, help="输入 ONNX 模型路径（sim 模型，batch=1）")
    parser.add_argument("--model_name", required=True, help="模型名称（用于输出文件命名）")
    parser.add_argument("--calibration_dir", required=True, help="ImageNet 校准数据目录")
    parser.add_argument("--save_dir", required=True, help="量化结果保存目录")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="原始模型 batch size（仅用于参数兼容，校准内部固定使用 32）",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=200,
        help="校准样本数（默认 200）",
    )
    parser.add_argument(
        "--calibrate_method",
        type=str,
        default="Entropy",
        choices=["Entropy", "MinMax", "Percentile"],
        help="校准方法（默认 Entropy：通过 KL 散度裁剪离群值，对 SE/Swish 激活效果最佳）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] 输入模型不存在: {args.input}")
        sys.exit(1)
    if not os.path.isdir(args.calibration_dir):
        print(f"[ERROR] 校准数据目录不存在: {args.calibration_dir}")
        sys.exit(1)
    os.makedirs(args.save_dir, exist_ok=True)

    # opset 版本检查（ORT 量化要求 opset >= 13）
    model = onnx.load(args.input)
    opset = model.opset_import[0].version
    if opset < 13:
        print(f"[INFO] opset={opset}，自动升级到 opset 13")
        model = onnx.version_converter.convert_version(model, 13)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.save(model, args.input)

    # 创建 batch=32 临时校准模型（修复 batch=1 校准时 SE/Swish 激活统计退化问题）
    # 参考 EfficientNet-V2-Tiny 的成功方案：sim.onnx batch=32 + batch_size=32 校准
    tmp_fd, tmp_calib_model = tempfile.mkstemp(suffix=".onnx")
    os.close(tmp_fd)
    try:
        print(f"[INFO] 创建 batch={_CALIB_BATCH_SIZE} 校准模型（修复 SE/Swish 激活统计问题）")
        create_batch32_model(args.input, tmp_calib_model, batch_size=_CALIB_BATCH_SIZE)

        output_path = os.path.join(args.save_dir, f"quantized_{args.model_name}.onnx")
        print(f"[INFO] 开始量化（仅 Conv，batch={_CALIB_BATCH_SIZE} 校准，"
              f"校准方法={args.calibrate_method}，样本数={args.num_samples}）")

        # 使用通用量化函数（含 ActivationSymmetric=True → zero_point=0，IxRT 兼容）
        # batch=32 校准确保 SE/Swish 激活统计正确，与 V2-Tiny 校准策略对齐
        # _NODES_TO_EXCLUDE 中的 13 个 Conv 节点以 FP16 运行，避免 SE 输出的对称量化精度损失
        quantize_with_torch_data(
            input_model=tmp_calib_model,
            output_model=output_path,
            calibration_dir=args.calibration_dir,
            num_samples=args.num_samples,
            batch_size=_CALIB_BATCH_SIZE,
            per_channel=True,
            op_types_to_quantize=["Conv"],
            nodes_to_exclude=_NODES_TO_EXCLUDE,
            calibrate_method=args.calibrate_method,
        )
    finally:
        if os.path.exists(tmp_calib_model):
            os.unlink(tmp_calib_model)

    # 清理冗余 axis 属性（IxRT 兼容性修复）
    remove_quantize_axis_attribute(output_path, output_path)
    print(f"[INFO] 量化完成，输出: {output_path}")


if __name__ == "__main__":
    main()