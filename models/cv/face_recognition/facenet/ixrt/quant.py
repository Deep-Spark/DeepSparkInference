import os
import random
import numpy as np
from pathlib import Path
from argparse import ArgumentParser

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import onnx
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


def fixed_image_standardization(image_tensor):
    return (image_tensor - 127.5) / 128.0


class FaceNetCalibrationReader(CalibrationDataReader):
    def __init__(self, dataloader, input_name):
        self.input_name = input_name
        self._iter = iter(dataloader)

    def get_next(self):
        try:
            data, _ = next(self._iter)
            if isinstance(data, torch.Tensor):
                return {self.input_name: data.cpu().numpy()}
            return None
        except StopIteration:
            return None


def create_dataloader(args):
    trans = transforms.Compose([
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization,
    ])
    dataset = datasets.ImageFolder(args.data_path + "lfw", transform=trans)
    print("image folder total images:", len(dataset))

    if args.num_samples is not None:
        indices = np.random.permutation(len(dataset))[:args.num_samples]
        calibration_dataset = torch.utils.data.Subset(dataset, indices=indices)
        print("calibration_dataset images:", len(calibration_dataset))
    else:
        calibration_dataset = dataset

    return DataLoader(
        calibration_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )


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


def create_argparser(*args, **kwargs):
    parser = ArgumentParser(*args, **kwargs)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--img_size", type=int, default=160)
    parser.add_argument("-j", "--workers", type=int, default=4)
    parser.add_argument("--model", type=str, default="./facenet_weights/facenet.onnx")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--data_path", type=str, default="./facenet_datasets/")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--observer", type=str, default="hist_percentile",
                        help="[unused] kept for CLI compatibility")
    parser.add_argument("--fp32_acc", action="store_true")
    parser.add_argument("--use_ixrt", action="store_true")
    parser.add_argument("--quant_params", type=str, default=None)
    parser.add_argument("--disable_bias_correction", action="store_true")
    return parser


def parse_args():
    parser = create_argparser("PTQ Quantization")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print(args)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    if args.model.endswith(".onnx"):
        model_name = os.path.basename(args.model).rsplit(".", maxsplit=1)[0]
    else:
        print("[Error] file name not correct:", args.model)
        return

    output_path = os.path.join("./facenet_weights", f"{model_name}-quant.onnx")

    onnx_model = onnx.load(args.model)
    if onnx_model.opset_import[0].version < 13:
        onnx_model = onnx.version_converter.convert_version(onnx_model, 13)
        onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
        onnx.save(onnx_model, args.model)

    input_name = onnx_model.graph.input[0].name
    dataloader = create_dataloader(args)
    calib_reader = FaceNetCalibrationReader(dataloader, input_name)

    quantize_static(
        model_input=args.model,
        model_output=output_path,
        calibration_data_reader=calib_reader,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        calibrate_method=CalibrationMethod.Entropy,
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