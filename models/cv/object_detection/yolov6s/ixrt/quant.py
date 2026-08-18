# Copyright (c) 2024, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.
"""Static INT8 quantization for detection models (letterbox-style).

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

from common import letterbox
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


def setseed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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
    """Make the batch dimension of all graph inputs dynamic (ORT calibration compat)."""
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
    parser.add_argument("--model", type=str, default="yolov4_bs16_without_decoder.onnx")
    parser.add_argument("--dataset_dir", type=str, default="./coco2017/val2017")
    parser.add_argument("--ann_file", type=str, default="./coco2017/annotations/instances_val2017.json")
    parser.add_argument("--observer", type=str, default="hist_percentile",
                        help="Calibration method: hist_percentile, percentile, entropy, minmax, ema")
    parser.add_argument("--disable_quant_names", nargs="*", type=str, default=None,
                        help="node names kept in float (passed to ORT nodes_to_exclude), "
                             "e.g. the detection head kept out of INT8 to match the "
                             "legacy tensorrt.deploy flow.")
    parser.add_argument("--save_quant_model", type=str, default=None,
                        help="path to write the quantized ONNX model")
    parser.add_argument("--bsz", type=int, default=16)
    parser.add_argument("--step", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--imgsz", type=int, default=608)
    parser.add_argument("--use_letterbox", action="store_true")
    return parser.parse_args()


def get_dataloader(data_dir, step=32, batch_size=16, new_shape=(608, 608), use_letterbox=False):
    num = step * batch_size
    val_list = [os.path.join(data_dir, x) for x in os.listdir(data_dir)]
    random.shuffle(val_list)
    pic_list = val_list[:num]

    calibration_dataset = []
    for file_path in pic_list:
        pic_data = cv2.imread(file_path)
        org_img = pic_data
        assert org_img is not None, 'Image not Found ' + file_path

        if use_letterbox:
            img, _, _ = letterbox(org_img, new_shape=(new_shape[1], new_shape[0]),
                                  auto=False, scaleup=True)
        else:
            img = cv2.resize(org_img, new_shape)
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img) / 255.0
        img = torch.from_numpy(img).float()

        calibration_dataset.append(img)

    calibration_dataloader = DataLoader(
        calibration_dataset,
        shuffle=True,
        batch_size=batch_size,
        drop_last=True,
    )
    return calibration_dataloader


def main():
    args = parse_args()
    setseed(args.seed)

    output_path = args.save_quant_model
    out_dir = os.path.dirname(output_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    ensure_opset13(args.model)

    dynamic_model_path = os.path.join(out_dir, f"_dynamic_{args.model_name}.onnx")
    make_input_dynamic(args.model, dynamic_model_path)

    dataloader = get_dataloader(
        data_dir=args.dataset_dir,
        step=args.step,
        batch_size=args.bsz,
        new_shape=(args.imgsz, args.imgsz),
        use_letterbox=args.use_letterbox,
    )

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
        nodes_to_exclude=(args.disable_quant_names or []),
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
