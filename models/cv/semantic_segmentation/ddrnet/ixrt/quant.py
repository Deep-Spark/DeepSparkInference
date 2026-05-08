import os
import cv2
import random
import argparse
import numpy as np
from pathlib import Path

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


# Patch: handle silent C++ failures in infer_shapes_path.
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


class DDRNetCalibrationReader(CalibrationDataReader):
    def __init__(self, dataloader, input_name):
        self.input_name = input_name
        self._iter = iter(dataloader)

    def get_next(self):
        try:
            batch = next(self._iter)
            if isinstance(batch, torch.Tensor):
                return {self.input_name: batch.cpu().numpy()}
            return None
        except StopIteration:
            return None


def getdataloader(datadir, list_path, step=32, batch_size=4):
    num = step * batch_size
    img_list = [line.strip().split()[0] for line in open(list_path)]
    val_list = [os.path.join(datadir, x) for x in img_list]
    random.shuffle(val_list)
    pic_list = val_list[:num]

    data = []
    for file_path in pic_list:
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        img = input_transform(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img = img.transpose((2, 0, 1))
        data.append(img)

    return DataLoader(data, shuffle=True, batch_size=batch_size, drop_last=True)


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ddrnet23.onnx")
    parser.add_argument("--dataset_dir", type=str, default="data/cityscapes")
    parser.add_argument("--list_path", type=str, default="data/list/cityscapes/val.lst")
    parser.add_argument("--save_dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    model_name = os.path.basename(args.model).rsplit(".", maxsplit=1)[0]
    out_dir = args.save_dir if args.save_dir else os.path.dirname(args.model)
    output_path = os.path.join(out_dir, f"quantized_{model_name}.onnx")

    model = onnx.load(args.model)
    if model.opset_import[0].version < 13:
        model = onnx.version_converter.convert_version(model, 13)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.save(model, args.model)

    input_name = model.graph.input[0].name
    dataloader = getdataloader(args.dataset_dir, args.list_path)
    calib_reader = DDRNetCalibrationReader(dataloader, input_name)

    quantize_static(
        model_input=args.model,
        model_output=output_path,
        calibration_data_reader=calib_reader,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        calibrate_method=CalibrationMethod.Percentile,
        extra_options={
            "ActivationSymmetric": True,
            "WeightSymmetric": True,
            "ZeroPoint": 0,
            "QuantizeBias": False,
            "EnableSubgraph": True,
        },
    )
    remove_quantize_axis_attribute(output_path, output_path)
    print(f"Quantization complete: {output_path}")


if __name__ == "__main__":
    main()