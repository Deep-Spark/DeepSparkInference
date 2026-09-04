"""YOLOv3 INT8 PTQ, calibrated to match the legacy ``tensorrt.deploy.static_quantize`` result.

Drop-in replacement for ``quant.py`` (same CLI) that closes the INT8 mAP@0.5 gap
between the two quantizers (legacy 0.657 vs ORT 0.649). Two differences account
for most of it:

1. ``hist_percentile`` was mapped to ``CalibrationMethod.Percentile`` but the
   percentile itself never got through. ``quantize_static`` only forwards four
   calibration options (``CalibTensorRangeSymmetric``, ``CalibMovingAverage``,
   ``CalibMovingAverageConstant``, ``CalibMaxIntermediateOutputs``), so a
   ``CalibPercentile`` entry in ``extra_options`` is silently dropped and the
   calibrator keeps its 99.999 default. At 99.999 almost the entire activation
   long tail is folded into the per-tensor scale, which is close to plain
   min-max; the legacy ixRT observer clips at 99.99. ``_calibrator_percentile``
   injects the value straight into ``create_calibrator``.

2. Legacy ``static_quantize`` ran bias correction by default
   (``qconfig.bias_correction = not disable_bias_correction``). ORT has no
   equivalent, so the per-channel bias introduced by weight quantization was
   left uncorrected. ``bias_correction`` below restores it.

Everything else (QDQ format, symmetric INT8, per-channel weights, float bias,
axis-attribute stripping) is kept identical to ``quant.py`` so the downstream
``deploy.py`` / ``modify_batchsize.py`` / ``build_engine.py`` steps are unaffected.
"""

import argparse
import importlib
import os
import random
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static

from calibration_dataset import create_dataloaders
from quant import (
    DetectionCalibrationReader,
    ensure_opset13,
    get_input_name,
    make_input_dynamic,
    remove_quantize_axis_attribute,
)

_OBSERVER_TO_CALIB_METHOD = {
    "hist_percentile": CalibrationMethod.Percentile,
    "percentile": CalibrationMethod.Percentile,
    "entropy": CalibrationMethod.Entropy,
    "minmax": CalibrationMethod.MinMax,
    "ema": CalibrationMethod.MinMax,
}

# Set by the QDQ quantizer on the int8 copy of a weight initializer.
_WEIGHT_QUANT_SUFFIX = "_quantized"

_BIAS_CORRECTION_OPS = ("Conv", "ConvTranspose")


@contextmanager
def _calibrator_percentile(percentile: float, num_bins: int):
    """Force the percentile through to the histogram calibrator.

    ``quantize_static`` filters ``extra_options`` down to a fixed set of calib
    keys that does not include the percentile, so patching ``create_calibrator``
    is the only way to set it without reimplementing the calibration loop.
    """
    # importlib, not a plain import: ``onnxruntime.quantization.quantize`` is
    # also re-exported as a function, which shadows the module attribute.
    ort_quantize = importlib.import_module("onnxruntime.quantization.quantize")

    original = ort_quantize.create_calibrator

    def patched(*args, **kwargs):
        extra = dict(kwargs.get("extra_options") or {})
        if kwargs.get("calibrate_method") == CalibrationMethod.Percentile:
            extra["percentile"] = percentile
            extra["num_bins"] = num_bins
        kwargs["extra_options"] = extra
        return original(*args, **kwargs)

    ort_quantize.create_calibrator = patched
    try:
        yield
    finally:
        ort_quantize.create_calibrator = original


@dataclass
class _CorrectionTarget:
    node_name: str
    op_type: str
    activation: str
    bias_name: str
    delta_weight: np.ndarray
    attributes: list


def _initializer_map(graph):
    return {init.name: init for init in graph.initializer}


def _dequantize_weight(dq_node, initializers):
    """Reconstruct the float weight the quantized Conv actually consumes."""
    q_name, scale_name = dq_node.input[0], dq_node.input[1]
    zp_name = dq_node.input[2] if len(dq_node.input) > 2 else None
    if q_name not in initializers or scale_name not in initializers:
        return None, None

    q = numpy_helper.to_array(initializers[q_name]).astype(np.float32)
    scale = numpy_helper.to_array(initializers[scale_name]).astype(np.float32)
    if zp_name in initializers:
        zp = numpy_helper.to_array(initializers[zp_name]).astype(np.float32)
    else:
        zp = np.zeros_like(scale)

    if scale.size > 1:
        axis = next((a.i for a in dq_node.attribute if a.name == "axis"), 0)
        broadcast = [1] * q.ndim
        broadcast[axis] = scale.size
        scale = scale.reshape(broadcast)
        zp = zp.reshape(broadcast)

    return (q - zp) * scale, q_name


def _collect_targets(quant_graph, float_initializers) -> List[_CorrectionTarget]:
    """Find INT8 Conv/ConvTranspose nodes that still carry a float bias."""
    producer = {out: node for node in quant_graph.node for out in node.output if out}
    initializers = _initializer_map(quant_graph)
    targets = []

    for node in quant_graph.node:
        if node.op_type not in _BIAS_CORRECTION_OPS or len(node.input) < 3:
            continue
        if node.input[2] not in initializers:
            # Bias was folded into a DequantizeLinear (QuantizeBias=True).
            continue

        dq = producer.get(node.input[1])
        if dq is None or dq.op_type != "DequantizeLinear":
            # Node was kept in float via nodes_to_exclude.
            continue

        weight_dq, q_name = _dequantize_weight(dq, initializers)
        if weight_dq is None:
            continue

        origin = q_name
        if origin.endswith(_WEIGHT_QUANT_SUFFIX):
            origin = origin[: -len(_WEIGHT_QUANT_SUFFIX)]
        if origin not in float_initializers:
            continue

        weight_float = numpy_helper.to_array(float_initializers[origin]).astype(np.float32)
        if weight_float.shape != weight_dq.shape:
            continue

        delta = weight_float - weight_dq
        if not np.any(delta):
            continue

        targets.append(
            _CorrectionTarget(
                node_name=node.name,
                op_type=node.op_type,
                activation=node.input[0],
                bias_name=node.input[2],
                delta_weight=delta,
                attributes=list(node.attribute),
            )
        )

    return targets


def _build_probe_model(quant_model, targets):
    """Expose each corrected node's input activation as a graph output."""
    probe = onnx.ModelProto()
    probe.CopyFrom(quant_model)
    existing = {out.name for out in probe.graph.output}
    for target in targets:
        if target.activation in existing:
            continue
        probe.graph.output.append(
            helper.make_tensor_value_info(target.activation, TensorProto.FLOAT, None)
        )
        existing.add(target.activation)
    return probe


def _build_delta_model(targets, ir_version):
    """Per-channel mean of ``conv(W_float - W_dequant, x)``.

    Convolution is linear in the weight, so with the same (already quantized)
    input activation this is exactly the legacy quantity
    ``fp_out - quant_out`` reduced over batch and spatial dims -- the layer's
    own weight-quantization error, without the upstream error that a naive
    float-vs-quantized output diff would double count.
    """
    nodes, inputs, outputs, initializers = [], [], [], []

    for idx, target in enumerate(targets):
        x_name, w_name = f"x_{idx}", f"w_{idx}"
        conv_out, mean_out = f"conv_{idx}", f"db_{idx}"

        inputs.append(helper.make_tensor_value_info(x_name, TensorProto.FLOAT, None))
        initializers.append(
            numpy_helper.from_array(target.delta_weight.astype(np.float32), w_name)
        )

        conv = helper.make_node(target.op_type, [x_name, w_name], [conv_out], name=f"delta_{idx}")
        conv.attribute.extend(target.attributes)
        nodes.append(conv)
        nodes.append(
            helper.make_node("ReduceMean", [conv_out], [mean_out], axes=[0, 2, 3], keepdims=0)
        )
        outputs.append(helper.make_tensor_value_info(mean_out, TensorProto.FLOAT, None))

    graph = helper.make_graph(nodes, "bias_correction_delta", inputs, outputs, initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    # onnx may default to an IR version newer than the runtime accepts; reuse
    # the one from the model we are correcting, which the runtime already loads.
    model.ir_version = ir_version
    return model


def bias_correction(quant_model_path, float_model_path, dataloader, input_name):
    """Add the mean per-channel weight-quantization error back into each bias."""
    quant_model = onnx.load(quant_model_path)
    float_model = onnx.load(float_model_path)

    targets = _collect_targets(quant_model.graph, _initializer_map(float_model.graph))
    del float_model
    if not targets:
        print("BiasCorrection: no eligible Conv nodes, skipped")
        return

    print(f"BiasCorrection: correcting {len(targets)} nodes")
    probe = _build_probe_model(quant_model, targets)
    probe_session = ort.InferenceSession(
        probe.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    delta_model = _build_delta_model(targets, quant_model.ir_version)
    delta_session = ort.InferenceSession(
        delta_model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    del probe

    activation_names = [t.activation for t in targets]
    error_sum = [np.zeros(t.delta_weight.shape[0], dtype=np.float64) for t in targets]
    samples = 0

    for batch in dataloader:
        data = batch[0] if isinstance(batch, (list, tuple)) else batch
        if not isinstance(data, torch.Tensor):
            continue
        images = data.cpu().numpy()

        activations = probe_session.run(activation_names, {input_name: images})
        errors = delta_session.run(
            None, {f"x_{i}": act for i, act in enumerate(activations)}
        )
        for i, error in enumerate(errors):
            error_sum[i] += np.asarray(error, dtype=np.float64) * len(images)
        samples += len(images)

    if samples == 0:
        print("BiasCorrection: calibration loader was empty, skipped")
        return

    initializers = _initializer_map(quant_model.graph)
    corrected = 0
    for target, total in zip(targets, error_sum):
        delta_bias = (total / samples).astype(np.float32)
        bias = numpy_helper.to_array(initializers[target.bias_name]).astype(np.float32)
        if bias.shape != delta_bias.shape:
            continue
        initializers[target.bias_name].CopyFrom(
            numpy_helper.from_array(bias + delta_bias, target.bias_name)
        )
        corrected += 1

    onnx.save(quant_model, quant_model_path)
    print(f"BiasCorrection: applied to {corrected}/{len(targets)} nodes over {samples} images")


def setseed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--model", type=str, default="yolov3_without_decoder.onnx")
    parser.add_argument("--data_process_type", type=str, default="yolov3")
    parser.add_argument("--dataset_dir", type=str, default="./coco2017/val2017")
    parser.add_argument("--ann_file", type=str, default="./coco2017/annotations/instances_val2017.json")
    parser.add_argument("--observer", type=str, default="hist_percentile",
                        help="Calibration method: hist_percentile, percentile, entropy, minmax, ema")
    parser.add_argument("--disable_quant_names", nargs="*", type=str, default=None,
                        help="node names kept in float (passed to ORT nodes_to_exclude)")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--bsz", type=int, default=32)
    parser.add_argument("--step", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--percentile", type=float, default=99.985,
                        help="YOLOv3 activation clipping percentile; 99.985 is "
                             "validated on COCO2017 (ORT defaults to 99.999)")
    parser.add_argument("--num_bins", type=int, default=2048,
                        help="calibration histogram bins, matching the legacy observer")
    parser.add_argument("--disable_bias_correction", action="store_true",
                        help="skip bias correction, which legacy static_quantize ran by default")
    return parser.parse_args()


def main():
    args = parse_args()
    setseed(args.seed)

    out_dir = args.save_dir
    output_path = os.path.join(out_dir, f"quantized_{args.model_name}.onnx")

    ensure_opset13(args.model)

    # ORT calibration rejects a batch size that disagrees with a hardcoded
    # static batch dim, so relax it before feeding calibration data.
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
    calib_method = _OBSERVER_TO_CALIB_METHOD.get(args.observer, CalibrationMethod.Percentile)
    print(f"Calibration: {calib_method}, percentile={args.percentile}, bins={args.num_bins}")

    with _calibrator_percentile(args.percentile, args.num_bins):
        quantize_static(
            model_input=dynamic_model_path,
            model_output=output_path,
            calibration_data_reader=DetectionCalibrationReader(dataloader, input_name),
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QInt8,
            quant_format=QuantFormat.QDQ,
            per_channel=True,
            calibrate_method=calib_method,
            nodes_to_exclude=(args.disable_quant_names or []),
            extra_options={
                "ActivationSymmetric": True,
                "WeightSymmetric": True,
                "ZeroPoint": 0,
                "QuantizeBias": False,
                "EnableSubgraph": True,
            },
        )

    if not args.disable_bias_correction:
        bias_correction(output_path, dynamic_model_path, dataloader, input_name)

    remove_quantize_axis_attribute(output_path, output_path)
    try:
        os.remove(dynamic_model_path)
    except OSError:
        pass
    print(f"Quantization complete: {output_path}")


if __name__ == "__main__":
    main()
