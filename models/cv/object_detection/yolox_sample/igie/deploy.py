# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Add IxRT custom YOLO decoder nodes to a backbone-only ONNX model.
Replaces the previous tensorrt.deploy.api-based implementation with
a pure onnx Python API approach that has no deploy dependency.

YOLOX variant: anchor-free decoder, takes one feature map per stride
(decoder_input has 3 entries -> num=1, decoder_32 consumes the trailing
slice ``decoder_input[num*2:]``).
"""
import argparse
import onnx
from onnx import helper, TensorProto


def _build_producer_map(graph):
    """name -> producing node. Initializers and graph-inputs map to None."""
    producer = {}
    for node in graph.node:
        for o in node.output:
            if o:
                producer[o] = node
    return producer


def remap_quant_inputs(graph, inputs):
    """Bypass any QDQ pair on each ``inputs`` tensor for IxRT plugin consumption."""
    producer = _build_producer_map(graph)
    remapped = []
    for name in inputs:
        prod = producer.get(name)
        if prod is not None and prod.op_type == "DequantizeLinear" and len(prod.input) >= 1:
            qdq_input = prod.input[0]
            qdq_prod = producer.get(qdq_input)
            if qdq_prod is not None and qdq_prod.op_type == "QuantizeLinear":
                remapped.append(qdq_input)
                continue
        if name not in producer:
            candidate = f"{name}_DequantizeLinear_Output"
            if candidate in producer:
                cand_prod = producer.get(candidate)
                if cand_prod is not None and cand_prod.op_type == "DequantizeLinear":
                    qdq_input = cand_prod.input[0]
                    qdq_prod = producer.get(qdq_input)
                    if qdq_prod is not None and qdq_prod.op_type == "QuantizeLinear":
                        remapped.append(qdq_input)
                        continue
                remapped.append(candidate)
                continue
        remapped.append(name)
    return remapped


def declare_value_info(graph, name, dtype=TensorProto.FLOAT):
    """Make ``name`` an explicit FLOAT value_info entry."""
    for vi in graph.value_info:
        if vi.name == name:
            return
    graph.value_info.append(helper.make_tensor_value_info(name, dtype, None))


def add_decoder_node(graph, op_type, inputs, outputs, anchor, num_class, stride, faster_impl):
    """Append an IxRT custom YOLO decoder node to the ONNX graph."""
    kwargs = dict(
        num_class=num_class,
        stride=stride,
        faster_impl=faster_impl,
    )
    # YoloxDecoder is anchor-free; keep the attribute out when the list is empty
    # (make_attribute raises ValueError on an empty list).
    if anchor:
        kwargs["anchor"] = [int(a) for a in anchor]
    node = helper.make_node(
        op_type,
        inputs=remap_quant_inputs(graph, inputs),
        outputs=outputs,
        name=f"node_of_{outputs[0]}",
        domain="",
        **kwargs,
    )
    graph.node.append(node)
    for out in outputs:
        declare_value_info(graph, out, TensorProto.FLOAT)


def add_concat_node(graph, inputs, outputs, axis=1):
    node = helper.make_node("Concat", inputs=inputs, outputs=outputs, axis=axis)
    graph.node.append(node)


def set_graph_outputs(graph, names):
    while len(graph.output) > 0:
        graph.output.pop()
    for name in names:
        graph.output.append(
            helper.make_tensor_value_info(name, TensorProto.FLOAT, None)
        )


def customize_ops(model, args):
    graph = model.graph
    decoder_input = args.decoder_input_names
    num = len(decoder_input) // 3

    add_decoder_node(
        graph, args.decoder_type,
        decoder_input[:num], ["decoder_8"],
        args.decoder8_anchor or [],
        args.num_class, 8, args.faster,
    )
    add_decoder_node(
        graph, args.decoder_type,
        decoder_input[num:num * 2], ["decoder_16"],
        args.decoder16_anchor or [],
        args.num_class, 16, args.faster,
    )
    # YOLOX: feed the remainder of decoder_input into decoder_32
    add_decoder_node(
        graph, args.decoder_type,
        decoder_input[num * 2:], ["decoder_32"],
        args.decoder32_anchor or [],
        args.num_class, 32, args.faster,
    )

    if args.decoder64_anchor is not None:
        add_decoder_node(
            graph, args.decoder_type,
            decoder_input[num * 2 + 1:], ["decoder_64"],
            args.decoder64_anchor or [],
            args.num_class, 64, args.faster,
        )
        add_concat_node(
            graph,
            ["decoder_8", "decoder_16", "decoder_32", "decoder_64"],
            ["output"], axis=1,
        )
        set_graph_outputs(graph, ["output"])
    elif args.with_nms:
        add_concat_node(
            graph,
            ["decoder_32", "decoder_16", "decoder_8"],
            ["output"], axis=1,
        )
        set_graph_outputs(graph, ["output"])
    else:
        set_graph_outputs(graph, ["decoder_8", "decoder_16", "decoder_32"])

    # Optional Focus replacement (FP16 / non-QDQ models only).
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
    parser.add_argument("--with_nms", type=bool, default=False, help="engine with nms")
    parser.add_argument("--decoder_input_names", nargs="+", type=str)
    parser.add_argument("--decoder8_anchor", nargs="*", type=int)
    parser.add_argument("--decoder16_anchor", nargs="*", type=int)
    parser.add_argument("--decoder32_anchor", nargs="*", type=int)
    parser.add_argument("--decoder64_anchor", nargs="*", type=int, default=None)
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
