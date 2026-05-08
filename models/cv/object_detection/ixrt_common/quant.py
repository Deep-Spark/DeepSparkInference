"""Static INT8 quantization for detection models.

Replaces the previous ``tensorrt.deploy.static_quantize`` flow with the
ONNX Runtime quantization API (``onnxruntime.quantization.quantize_static``)
so that this script no longer depends on the IxRT private deploy package.
"""
import os
import argparse
import random
from pathlib import Path

import numpy as np
import onnx
import torch

from calibration_dataset import create_dataloaders
from onnxruntime.quantization import (
    quantize_static,
    QuantType,
    CalibrationDataReader,
    QuantFormat,
    CalibrationMethod,
)


_OBSERVER_TO_CALIB_METHOD = {
    "hist_percentile": CalibrationMethod.Percentile,
    "percentile":      CalibrationMethod.Percentile,
    "entropy":         CalibrationMethod.Entropy,
    "minmax":          CalibrationMethod.MinMax,
    "ema":             CalibrationMethod.MinMax,
}


# Patch: handle silent C++ failures in infer_shapes_path (e.g. for dynamic-batch models).
# When infer_shapes_path succeeds, the patch is transparent;
# otherwise fall back to in-memory shape inference.
_orig_infer_shapes_path = onnx.shape_inference.infer_shapes_path


def _robust_infer_shapes_path(input_path: str, output_path: str, **kwargs):
    try:
        _orig_infer_shapes_path(input_path, output_path, **kwargs)
    except Exception:
        pass
    if not Path(output_path).exists():
        model = onnx.load(input_path)
        inferred = onnx.shape_inference.infer_shapes(model)
        onnx.save(inferred, output_path)


onnx.shape_inference.infer_shapes_path = _robust_infer_shapes_path


from onnxruntime.quantization import (
    quantize_static,
    QuantType,
    CalibrationDataReader,
    QuantFormat,
    CalibrationMethod,
)

_OBSERVER_TO_CALIB_METHOD = {
    "hist_percentile": CalibrationMethod.Percentile,
    "percentile":      CalibrationMethod.Percentile,
    "entropy":         CalibrationMethod.Entropy,
    "minmax":          CalibrationMethod.MinMax,
    "ema":             CalibrationMethod.MinMax,
}


# Patch: handle silent C++ failures in infer_shapes_path (e.g. for dynamic-batch models).
_orig_infer_shapes_path = onnx.shape_inference.infer_shapes_path
def _robust_infer_shapes_path(input_path: str, output_path: str, **kwargs):
    try:
        _orig_infer_shapes_path(input_path, output_path, **kwargs)
    except Exception:
        pass
    if not Path(output_path).exists():
        model = onnx.load(input_path)
        inferred = onnx.shape_inference.infer_shapes(model)
        onnx.save(inferred, output_path)
onnx.shape_inference.infer_shapes_path = _robust_infer_shapes_path


class DetectionCalibrationReader(CalibrationDataReader):
    def __init__(self, dataloader, input_name):
        self.dataloader = dataloader
        self.input_name = input_name
        self._iter = iter(dataloader)

    def get_next(self):
        try:
            batch = next(self._iter)
            # CocoDetection returns (image, origin_shape, image_id) → batched as 3-tuple
            data = batch[0] if isinstance(batch, (list, tuple)) else batch
            if isinstance(data, torch.Tensor):
                return {self.input_name: data.cpu().numpy()}
            return None
        except StopIteration:
            return None


def remove_quantize_axis_attribute(model_path, output_path):
    model = onnx.load(model_path)
    quantize_node_types = {"QuantizeLinear", "DequantizeLinear", "DynamicQuantizeLinear"}
    init_names = {init.name for init in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type not in quantize_node_types:
            continue
        if (node.op_type == "DequantizeLinear"
                and len(node.input) > 0
                and node.input[0] in init_names):
            continue
        to_remove = [i for i, attr in enumerate(node.attribute) if attr.name == "axis"]
        for i in reversed(to_remove):
            del node.attribute[i]
    onnx.save(model, output_path)


def get_input_name(model_path):
    model = onnx.load(model_path)
    return model.graph.input[0].name


def make_input_dynamic(model_path: str, output_path: str) -> None:
    """Make the batch dimension of all graph inputs dynamic.

    Some without_decoder ONNX files have a hardcoded static batch size (e.g. 16).
    ORT calibration requires the actual batch size to match the model's static size.
    Making the batch dim dynamic allows ORT to accept any batch size during calibration.
    """
    model = onnx.load(model_path)
    for inp in model.graph.input:
        if inp.type.HasField("tensor_type") and inp.type.tensor_type.HasField("shape"):
            dims = inp.type.tensor_type.shape.dim
            if len(dims) > 0 and dims[0].dim_value != 1:
                dims[0].ClearField("dim_value")
                dims[0].dim_param = "N"
    onnx.save(model, output_path)


def ensure_opset13(model_path):
    model = onnx.load(model_path)
    if model.opset_import[0].version < 13:
        model = onnx.version_converter.convert_version(model, 13)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.save(model, model_path)



class DetectionCalibrationReader(CalibrationDataReader):
    def __init__(self, dataloader, input_name):
        self.dataloader = dataloader
        self.input_name = input_name
        self._iter = iter(dataloader)

    def get_next(self):
        try:
            batch = next(self._iter)
            # CocoDetection returns (image, origin_shape, image_id) - batched as a tuple.
            data = batch[0] if isinstance(batch, (list, tuple)) else batch
            if isinstance(data, torch.Tensor):
                return {self.input_name: data.cpu().numpy()}
            return None
        except StopIteration:
            return None


def remove_quantize_axis_attribute(model_path, output_path):
    """Strip the ``axis`` attribute from Q/DQ activation nodes (IxRT compatibility)."""
    model = onnx.load(model_path)
    quantize_node_types = {"QuantizeLinear", "DequantizeLinear", "DynamicQuantizeLinear"}
    init_names = {init.name for init in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type not in quantize_node_types:
            continue
        # Keep ``axis`` on weight-side DequantizeLinear (per-channel DQ).
        if (node.op_type == "DequantizeLinear"
                and len(node.input) > 0
                and node.input[0] in init_names):
            continue
        to_remove = [i for i, attr in enumerate(node.attribute) if attr.name == "axis"]
        for i in reversed(to_remove):
            del node.attribute[i]
    onnx.save(model, output_path)


def get_input_name(model_path):
    model = onnx.load(model_path)
    return model.graph.input[0].name


def make_input_dynamic(model_path: str, output_path: str) -> None:
    """Make the batch dimension of all graph inputs dynamic.

    Some without_decoder ONNX files have a hardcoded static batch size (e.g. 16).
    ORT calibration requires the actual batch size to match the model's static
    size; making the batch dim dynamic allows ORT to accept any batch size.
    """
    model = onnx.load(model_path)
    for inp in model.graph.input:
        if inp.type.HasField("tensor_type") and inp.type.tensor_type.HasField("shape"):
            dims = inp.type.tensor_type.shape.dim
            if len(dims) > 0 and dims[0].dim_value != 1:
                dims[0].ClearField("dim_value")
                dims[0].dim_param = "N"
    onnx.save(model, output_path)


def ensure_opset13(model_path):
    """QDQ-format quantization requires opset >= 13."""
    model = onnx.load(model_path)
    if model.opset_import[0].version < 13:
        model = onnx.version_converter.convert_version(model, 13)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.save(model, model_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--model", type=str, default="yolov5s_with_decoder.onnx")
    parser.add_argument("--data_process_type", type=str, default="none")
    parser.add_argument("--dataset_dir", type=str, default="./coco2017/val2017")
    parser.add_argument("--ann_file", type=str, default="./coco2017/annotations/instances_val2017.json")
    parser.add_argument("--observer", type=str, default="hist_percentile",
                        help="Calibration method: hist_percentile, percentile, entropy, minmax, ema")
    parser.add_argument("--disable_quant_names", nargs="*", type=str,
                        help="[unused] kept for CLI compatibility")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--bsz", type=int, default=32)
    parser.add_argument("--step", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args()


def main():
    args = parse_args()
    setseed(args.seed)

    out_dir = args.save_dir
    output_path = os.path.join(out_dir, f"quantized_{args.model_name}.onnx")

    ensure_opset13(args.model)

    # Make batch dimension dynamic for ORT calibration compatibility.
    dynamic_model_path = os.path.join(out_dir, f"_dynamic_{args.model_name}_without_decoder.onnx")
    make_input_dynamic(args.model, dynamic_model_path)

    dataloader = create_dataloaders(
        data_path=args.dataset_dir,
        annFile=args.ann_file,
        img_sz=args.imgsz,
        batch_size=args.bsz,
        step=args.step,
        data_process_type=args.data_process_type,
    )

    input_name = get_input_name(args.model)
    calib_reader = DetectionCalibrationReader(dataloader, input_name)

    calib_method = _OBSERVER_TO_CALIB_METHOD.get(args.observer, CalibrationMethod.Percentile)

    extra_options = {
        "ActivationSymmetric": True,
        "WeightSymmetric": True,
        "ZeroPoint": 0,
        "QuantizeBias": False,
        "EnableSubgraph": True,
    }
    # Calibration-method specific tuning to better match the precision of the
    # original ``tensorrt.deploy.static_quantize`` (hist_percentile) flow:
    # ``CalibPercentile=99.99`` (vs ORT default 99.999) keeps long-tail
    # activation outliers from inflating the per-tensor scale, which the IxRT
    # hist_percentile observer also clips by default.
    if calib_method == CalibrationMethod.Percentile:
        extra_options["CalibPercentile"] = 99.99
    quantize_static(
        model_input=dynamic_model_path,
        model_output=output_path,
        calibration_data_reader=calib_reader,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        calibrate_method=calib_method,
        extra_options=extra_options,
    )
    remove_quantize_axis_attribute(output_path, output_path)
    try:
        os.remove(dynamic_model_path)
    except OSError:
        pass
    print(f"Quantization complete: {output_path}")


if __name__ == "__main__":
    main()
