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


def _build_producer_map(graph):
    """name -> producing node. Initializers and graph-inputs map to None."""
    producer = {}
    for node in graph.node:
        for o in node.output:
            if o:
                producer[o] = node
    return producer


def remap_quant_inputs(graph, inputs):
    """Bypass any QDQ pair on each ``inputs`` tensor for IxRT plugin consumption.

    IxRT's YoloV*Decoder plugin (and its INT8 fast path) expects to be fed the
    INT8 ``QuantizeLinear`` output, not the dequantized FLOAT tensor. After
    ORT's ``quantize_static`` the typical layout is::

        Conv -> <X>_QuantizeLinear_Input
             -> QuantizeLinear -> <X>_QuantizeLinear_Output (INT8)
             -> DequantizeLinear -> <X> (FLOAT)  <-- we get this name

    so when we detect that an input is the output of a DequantizeLinear, we
    walk back one step and use the upstream ``QuantizeLinear`` output instead.
    Tensors that are not part of a QDQ pair (e.g. FP16 models) pass through
    unchanged.
    """
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
        # Fallback: ORT-only renaming (DQ output suffix) when the raw name was
        # passed but is no longer present in the graph.
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
    """Make ``name`` an explicit FLOAT value_info entry.

    IxRT's algorithm-selection pass needs the output dtype of custom plugins
    (e.g. YoloV*Decoder) to be explicit; without a value_info entry it cannot
    infer the FLOAT output type and the format-combination candidate set
    collapses to empty, surfacing as
    "fail to find satisfied combinations ... io_idx <1>".
    """
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
    # Only include anchor when non-empty; anchor-free decoders (e.g. YoloxDecoder)
    # must omit this attribute -- make_attribute raises ValueError on an empty list.
    if anchor:
        # Cast to int so onnx encodes ``anchor`` as INTS (the IxRT YOLO decoder
        # plugin reads anchors as integers). Tolerates float CLI inputs.
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
    """Replace the graph outputs with the given list of FLOAT tensor names."""
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

    if args.decoder64_anchor is not None:
        add_decoder_node(
            graph, args.decoder_type,
            decoder_input[num * 2:num * 2 + 1], ["decoder_32"],
            args.decoder32_anchor or [],
            args.num_class, 32, args.faster,
        )
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
        concat_inputs = ["decoder_8", "decoder_16", "decoder_32", "decoder_64"]
    else:
        add_decoder_node(
            graph, args.decoder_type,
            decoder_input[num * 2:], ["decoder_32"],
            args.decoder32_anchor or [],
            args.num_class, 32, args.faster,
        )
        add_concat_node(
            graph,
            ["decoder_32", "decoder_16", "decoder_8"],
            ["output"], axis=1,
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

    set_graph_outputs(graph, ["output"])

    # Optional Focus replacement (FP16 / non-QDQ models only).
    # For INT8 QDQ models the original Focus subgraph would be dead-code-
    # eliminated by IxRT before the custom Focus node could resolve its input,
    # so we only insert the custom Focus op when the input is the raw graph
    # tensor (e.g. "images"), not a DequantizeLinear output.
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
