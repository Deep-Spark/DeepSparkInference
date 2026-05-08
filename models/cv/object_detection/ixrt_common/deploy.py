# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Add IxRT custom YOLO decoder nodes to a backbone-only ONNX model.
Replaces the previous tensorrt.deploy.api-based implementation with
a pure onnx Python API approach that has no deploy dependency.
"""
import argparse
import onnx
from onnx import helper, TensorProto


def add_decoder_node(graph, op_type, inputs, outputs, anchor, num_class, stride, faster_impl):
    """Append an IxRT custom decoder node to the ONNX graph."""
    kwargs = dict(
        num_class=num_class,
        stride=stride,
        faster_impl=faster_impl,
    )
    # Only include anchor when non-empty; anchor-free decoders (e.g. YoloxDecoder)
    # must omit this attribute — make_attribute raises ValueError on an empty list.
    if anchor:
        kwargs["anchor"] = anchor
    node = helper.make_node(
        op_type,
        inputs=inputs,
        outputs=outputs,
        domain="",
        **kwargs,
    )
    graph.node.append(node)


def customize_ops(model, args):
    graph = model.graph
    decoder_input = args.decoder_input_names
    num = len(decoder_input) // 3

    add_decoder_node(
        graph,
        op_type=args.decoder_type,
        inputs=decoder_input[:num],
        outputs=["decoder_8"],
        anchor=args.decoder8_anchor if args.decoder8_anchor else [],
        num_class=args.num_class,
        stride=8,
        faster_impl=args.faster,
    )
    add_decoder_node(
        graph,
        op_type=args.decoder_type,
        inputs=decoder_input[num:num * 2],
        outputs=["decoder_16"],
        anchor=args.decoder16_anchor if args.decoder16_anchor else [],
        num_class=args.num_class,
        stride=16,
        faster_impl=args.faster,
    )

    if args.decoder64_anchor is not None:
        add_decoder_node(
            graph,
            op_type=args.decoder_type,
            inputs=decoder_input[num * 2:num * 2 + 1],
            outputs=["decoder_32"],
            anchor=args.decoder32_anchor if args.decoder32_anchor else [],
            num_class=args.num_class,
            stride=32,
            faster_impl=args.faster,
        )
        add_decoder_node(
            graph,
            op_type=args.decoder_type,
            inputs=decoder_input[num * 2 + 1:],
            outputs=["decoder_64"],
            anchor=args.decoder64_anchor,
            num_class=args.num_class,
            stride=64,
            faster_impl=args.faster,
        )
        concat_inputs = ["decoder_8", "decoder_16", "decoder_32", "decoder_64"]
    else:
        add_decoder_node(
            graph,
            op_type=args.decoder_type,
            inputs=decoder_input[num * 2:],
            outputs=["decoder_32"],
            anchor=args.decoder32_anchor if args.decoder32_anchor else [],
            num_class=args.num_class,
            stride=32,
            faster_impl=args.faster,
        )
        concat_inputs = ["decoder_32", "decoder_16", "decoder_8"]

    concat_node = helper.make_node(
        "Concat",
        inputs=concat_inputs,
        outputs=["output"],
        axis=1,
    )
    graph.node.append(concat_node)

    # Rewire graph outputs to the single fused "output" tensor
    while len(graph.output) > 0:
        graph.output.pop()
    graph.output.append(
        helper.make_tensor_value_info("output", TensorProto.FLOAT, None)
    )

    # Optional Focus replacement (FP16 / non-QDQ models only).
    # For INT8 QDQ models the input tensor is named
    # "images_DequantizeLinear_Output" (the DQ output after the input QDQ pair).
    # In that case IxRT dead-code-eliminates the original Focus subgraph when
    # its output is renamed, which causes "images_DequantizeLinear_Output" to
    # be unregistered by the time the custom Focus node is parsed.
    # Therefore we only add the custom Focus op for FP16 models where
    # focus_input is the raw graph-input tensor (e.g. "images").
    if (args.focus_input is not None
            and args.focus_output is not None
            and not args.focus_input.endswith("_DequantizeLinear_Output")):
        focus_output_names = (
            args.focus_output
            if isinstance(args.focus_output, list)
            else [args.focus_output]
        )
        focus_node = helper.make_node(
            "Focus",
            inputs=[args.focus_input],
            outputs=focus_output_names,
            domain="",
        )
        graph.node.append(focus_node)

    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str)
    parser.add_argument("--dst", type=str)
    parser.add_argument("--decoder_type", type=str,
                        choices=["YoloV3Decoder", "YoloV5Decoder", "YoloV7Decoder", "YoloxDecoder"])
    parser.add_argument("--decoder_input_names", nargs="+", type=str)
    parser.add_argument("--decoder8_anchor", nargs="*", type=float)
    parser.add_argument("--decoder16_anchor", nargs="*", type=float)
    parser.add_argument("--decoder32_anchor", nargs="*", type=float)
    parser.add_argument("--decoder64_anchor", nargs="*", type=float, default=None)
    parser.add_argument("--num_class", type=int, default=80)
    parser.add_argument("--faster", type=int, default=1)
    parser.add_argument("--focus_input", type=str, default=None)
    parser.add_argument("--focus_output", nargs="*", type=str, default=None)
    parser.add_argument("--focus_last_node", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model = onnx.load(args.src)
    model = customize_ops(model, args)
    onnx.save(model, args.dst)
    print("Surged onnx lies on", args.dst)