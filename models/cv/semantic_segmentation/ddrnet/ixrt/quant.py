"""Static INT8 quantization for DDRNet (Cityscapes calibration).

Replaces the previous ``tensorrt.deploy.static_quantize`` flow with the
ONNX Runtime quantization API (``onnxruntime.quantization.quantize_static``)
so that this script no longer depends on the IxRT private deploy package.
"""
import os
import cv2
import random
import argparse
from pathlib import Path

import numpy as np
import onnx
import torch
from torch.utils.data import DataLoader

from utils import input_transform
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ddrnet23.onnx")
    parser.add_argument("--dataset_dir", type=str, default="/root/data/datasets/cityscapes")
    parser.add_argument("--list_path", type=str, default="/root/data/datasets/cityscapes/val.lst",
                        help="The path of val list.")
    parser.add_argument("--save_dir", type=str, default=None, help="quant file")
    parser.add_argument("--observer", type=str, default="percentile",
                        help="Calibration method: hist_percentile, percentile, entropy, minmax, ema")
    parser.add_argument("--bsz", type=int, default=4)
    parser.add_argument("--step", type=int, default=32)
    return parser.parse_args()


class TensorCalibrationReader(CalibrationDataReader):
    def __init__(self, dataloader, input_name):
        self.dataloader = dataloader
        self.input_name = input_name
        self._iter = iter(dataloader)

    def get_next(self):
        try:
            batch = next(self._iter)
            data = batch[0] if isinstance(batch, (list, tuple)) else batch
            if isinstance(data, torch.Tensor):
                return {self.input_name: data.cpu().numpy()}
            return None
        except StopIteration:
            return None


def getdataloader(datadir, list_path, step=32, batch_size=4):
    num = step * batch_size

    img_list = [line.strip().split()[0] for line in open(list_path)]
    val_list = [os.path.join(datadir, x) for x in img_list]
    random.shuffle(val_list)
    pic_list = val_list[:num]

    dataset = []
    for file_path in pic_list:
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        img = input_transform(
            img,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        img = img.transpose((2, 0, 1))
        dataset.append(img)

    calibration_dataloader = DataLoader(
        dataset,
        shuffle=True,
        batch_size=batch_size,
        drop_last=True,
    )
    return calibration_dataloader


def remove_quantize_axis_attribute(model_path, output_path):
    """Strip the ``axis`` attribute from Q/DQ activation nodes (IxRT compatibility)."""
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


def main():
    args = parse_args()
    model_name = os.path.basename(args.model)
    model_name = model_name.rsplit(".", maxsplit=1)[0]

    out_dir = args.save_dir or os.path.dirname(args.model) or "."
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, f"quantized_{model_name}.onnx")

    ensure_opset13(args.model)

    dynamic_model_path = os.path.join(out_dir, f"_dynamic_{model_name}.onnx")
    make_input_dynamic(args.model, dynamic_model_path)

    dataloader = getdataloader(args.dataset_dir, args.list_path,
                               step=args.step, batch_size=args.bsz)

    input_name = get_input_name(args.model)
    calib_reader = TensorCalibrationReader(dataloader, input_name)

    calib_method = _OBSERVER_TO_CALIB_METHOD.get(args.observer, CalibrationMethod.Percentile)

    extra_options = {
        "ActivationSymmetric": True,
        "WeightSymmetric": True,
        "ZeroPoint": 0,
        "QuantizeBias": False,
        "EnableSubgraph": True,
    }
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
