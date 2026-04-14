import os
import random
import argparse
import numpy as np

import onnx
if not hasattr(onnx, "mapping"):
    import types
    onnx.mapping = types.ModuleType("onnx.mapping")
    onnx.mapping.TENSOR_TYPE_TO_NP_TYPE = {
        k: v.np_dtype for k, v in onnx._mapping.TENSOR_TYPE_MAP.items()
    }
import torch
from onnxruntime.quantization import (
    quantize_static,
    QuantType,
    CalibrationDataReader,
    QuantFormat,
    CalibrationMethod,
)
from calibration_dataset import getdataloader

OBSERVER_MAP = {
    "minmax": CalibrationMethod.MinMax,
    "entropy": CalibrationMethod.Entropy,
    "percentile": CalibrationMethod.Percentile,
    "hist_percentile": CalibrationMethod.Percentile,
    "ema": CalibrationMethod.MinMax,
}


class TorchCalibrationDataReader(CalibrationDataReader):
    def __init__(self, dataloader, input_name):
        self.dataloader = dataloader
        self.input_name = input_name
        self.iterator = iter(dataloader)

    def get_next(self):
        try:
            data, _ = next(self.iterator)
            if isinstance(data, torch.Tensor):
                return {self.input_name: data.cpu().numpy()}
            return None
        except StopIteration:
            return None


def fix_quantize_axis_attribute(model_path, output_path):
    """Reset axis to 0 on all QuantizeLinear/DequantizeLinear nodes.

    ONNX opset>=13 defaults axis to 1 when the attribute is absent,
    which is out of range for 1-D tensors like biases.  Setting axis=0
    is safe for both per-tensor (scalar scale/zp) and per-channel cases.
    """

    model = onnx.load(model_path)
    quantize_node_types = {"QuantizeLinear", "DequantizeLinear", "DynamicQuantizeLinear"}
    for node in model.graph.node:
        if node.op_type in quantize_node_types:
            found = False
            for attr in node.attribute:
                if attr.name == "axis":
                    attr.i = 0
                    found = True
            if not found:
                node.attribute.append(onnx.helper.make_attribute("axis", 0))
    onnx.save(model, output_path)


def get_onnx_input_name(model_path):
    model = onnx.load(model_path)
    return model.graph.input[0].name


def setseed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--dataset_dir", type=str, default="imagenet_val")
    parser.add_argument("--observer", type=str,
                        choices=["hist_percentile", "percentile", "minmax", "entropy", "ema"],
                        default="hist_percentile")
    parser.add_argument("--disable_quant_names", nargs='*', type=str)
    parser.add_argument("--save_dir", type=str, help="save path", default=None)
    parser.add_argument("--bsz", type=int, default=32)
    parser.add_argument("--step", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--imgsz", type=int, default=224)
    args = parser.parse_args()
    print("Quant config:", args)
    print(args.disable_quant_names)
    return args


def main():
    args = parse_args()
    setseed(args.seed)

    model = onnx.load(args.model)
    opset = model.opset_import[0].version
    if opset < 13:
        print(f"opset version: {opset}, converting to 13 for QDQ quantization")
        model = onnx.version_converter.convert_version(model, 13)
        onnx.save(model, args.model)

    input_name = get_onnx_input_name(args.model)
    print(f"Model input name: {input_name}")

    calibration_dataloader = getdataloader(
        args.dataset_dir, args.step, args.bsz, img_sz=args.imgsz
    )
    calib_reader = TorchCalibrationDataReader(calibration_dataloader, input_name)

    output_path = os.path.join(args.save_dir, f"quantized_{args.model_name}.onnx")
    calibrate_method = OBSERVER_MAP.get(args.observer, CalibrationMethod.MinMax)

    nodes_to_exclude = args.disable_quant_names or []

    print(f"Calibration method: {calibrate_method}")
    print(f"Nodes to exclude: {nodes_to_exclude}")

    quantize_static(
        model_input=args.model,
        model_output=output_path,
        calibration_data_reader=calib_reader,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        calibrate_method=calibrate_method,
        nodes_to_exclude=nodes_to_exclude,
        extra_options={
            "ActivationSymmetric": True,
            "WeightSymmetric": True,
            "QuantizeBias": False,
        },
    )
    fix_quantize_axis_attribute(output_path, output_path)
    print(f"Quantized model saved to: {output_path}")


if __name__ == "__main__":
    main()
