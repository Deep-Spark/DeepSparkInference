"""Static INT8 quantization for FaceNet (LFW calibration set).

Replaces the previous ``tensorrt.deploy.static_quantize`` flow with the
ONNX Runtime quantization API (``onnxruntime.quantization.quantize_static``)
so that this script no longer depends on the IxRT private deploy package.
"""
import os
import random
import json
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import onnx
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
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


device = 0 if torch.cuda.is_available() else "cpu"


def setseed(seed=43):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def fixed_image_standardization(image_tensor):
    processed_tensor = (image_tensor - 127.5) / 128.0
    return processed_tensor


class FaceCalibrationReader(CalibrationDataReader):
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


def create_dataloader(args):
    image_dir_path = os.path.join(args.data_path, "lfw")

    trans = transforms.Compose([
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization,
    ])

    dataset = datasets.ImageFolder(image_dir_path, transform=trans)

    calibration_dataset = dataset
    print("image folder total images : ", len(dataset))
    if args.num_samples is not None:
        indices = np.random.permutation(len(dataset))[:args.num_samples]
        calibration_dataset = torch.utils.data.Subset(dataset, indices=indices)
        print("calibration_dataset images : ", len(calibration_dataset))

    assert len(dataset), "data size is 0, check data path please"
    calibration_dataloader = DataLoader(
        calibration_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )
    verify_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )

    return calibration_dataloader, verify_dataloader


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


@torch.no_grad()
def quantize_model(args, model_name, model_path, dataloader):
    calibration_dataloader, _ = dataloader
    print("calibration dataset length: ", len(calibration_dataloader))

    out_dir = "./facenet_weights"
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, f"{model_name}-quant.onnx")

    ensure_opset13(model_path)

    dynamic_model_path = os.path.join(out_dir, f"_dynamic_{model_name}.onnx")
    make_input_dynamic(model_path, dynamic_model_path)

    input_name = get_input_name(model_path)
    calib_reader = FaceCalibrationReader(calibration_dataloader, input_name)

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


def add_1190_scale(cfg_name):
    """Legacy: copy the quant scale of tensor 1189 onto 1190 in the IxRT params JSON.

    The previous ``static_quantize`` flow produced a side-car JSON describing
    per-tensor scales; ORT QDQ embeds those scales as graph initializers
    instead, so this helper is a no-op when the JSON is absent.
    """
    if not os.path.isfile(cfg_name):
        return
    graph_json = json.load(open(cfg_name))
    if "quant_info" in graph_json and "1189" in graph_json["quant_info"]:
        graph_json["quant_info"]["1190"] = graph_json["quant_info"]["1189"]
        with open(cfg_name, "w") as fh:
            json.dump(graph_json, fh, indent=4)


def create_argparser(*args, **kwargs):
    parser = ArgumentParser(*args, **kwargs)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--img_size", type=int, default=160)
    parser.add_argument("-j", "--workers", type=int, default=4)
    parser.add_argument("--model", type=str, default="./facenet_weights/facenet.onnx")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--data_path", type=str, default="./facenet_datasets/")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--observer", type=str, default="hist_percentile")
    parser.add_argument("--fp32_acc", action="store_true")
    parser.add_argument("--use_ixrt", action="store_true")
    parser.add_argument("--quant_params", type=str, default=None)
    parser.add_argument("--disable_bias_correction", action="store_true")
    return parser


def parse_args():
    parser = create_argparser("PTQ Quantization")
    args = parser.parse_args()
    args.use_ixquant = not args.use_ixrt
    return args


def main():
    setseed()
    args = parse_args()
    print(args)
    dataloader = create_dataloader(args)

    if args.model.endswith(".onnx"):
        model_name = os.path.basename(args.model)
        model_name = model_name.rsplit(".", maxsplit=1)[0]
        model_path = args.model
    else:
        print("[Error] file name not correct ", args.model)
        return
    quantize_model(args, model_name, model_path, dataloader)
    add_1190_scale(f"./facenet_weights/{model_name}.json")


if __name__ == "__main__":
    main()
