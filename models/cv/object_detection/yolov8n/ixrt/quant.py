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
    """Rewrite the ONNX so the batch dimension is fully dynamic.

    ultralytics exports YOLO with a *static* batch, baking it not only into the
    input shape but also into the detection-head ``Reshape`` targets (e.g.
    ``[32, 64, -1]``, ``[32, 4, 16, 8400]``). Two problems follow:

      * calibration is forced to use that same large batch, and ORT's
        hist_percentile keeps full activations per batch -> memory blows up and
        the test machine OOMs at bs=32/64;
      * even a dynamic engine cannot infer at other batch sizes because the
        Reshape batch is hard-coded.

    Making the batch truly dynamic decouples calibration from inference: we can
    calibrate at bsz=1 (low memory, batch-agnostic Q/DQ scales) while the engine
    still serves any batch. We:
      * turn the input/output batch dim into a symbol;
      * set every Reshape target's leading dim that equals the static batch to
        ``0`` (ONNX "copy the dimension from the input at runtime").
    """
    from onnx import numpy_helper

    model = onnx.load(model_path)

    static_batch = None
    in_dims = model.graph.input[0].type.tensor_type.shape.dim
    if len(in_dims) > 0 and in_dims[0].dim_value > 0:
        static_batch = in_dims[0].dim_value

    for tensor in list(model.graph.input) + list(model.graph.output):
        if tensor.type.HasField("tensor_type") and tensor.type.tensor_type.HasField("shape"):
            dims = tensor.type.tensor_type.shape.dim
            if len(dims) > 0:
                dims[0].ClearField("dim_value")
                dims[0].dim_param = "N"

    if static_batch is not None:
        inits = {init.name: init for init in model.graph.initializer}
        for node in model.graph.node:
            if node.op_type != "Reshape" or len(node.input) < 2:
                continue
            init = inits.get(node.input[1])
            if init is None:
                continue
            arr = numpy_helper.to_array(init).copy()
            if arr.size > 0 and arr[0] == static_batch:
                arr[0] = 0  # copy batch from the input tensor at runtime
                init.CopyFrom(numpy_helper.from_array(arr, init.name))

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
                        help="node names kept in float (passed to ORT nodes_to_exclude). "
                             "Used to keep the YOLOv8 detection head (DFL softmax, box "
                             "decode, sigmoid) out of INT8, matching the legacy deploy flow.")
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

    # Keep the sensitive detection head in float. The old tensorrt.deploy flow
    # skipped these nodes; quantizing DFL softmax / box-decode arithmetic /
    # sigmoid to INT8 collapses mAP to ~0. ORT honours this via nodes_to_exclude.
    nodes_to_exclude = args.disable_quant_names or []

    quantize_static(
        model_input=dynamic_model_path,
        model_output=output_path,
        calibration_data_reader=calib_reader,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        calibrate_method=calib_method,
        nodes_to_exclude=nodes_to_exclude,
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
