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


def replace_focus(graph, focus_output):
    """Fuse the YOLOX Focus slice/concat subgraph into a single IxRT ``Focus`` op.

    The Focus module slices the input image (space-to-depth) and concatenates
    the 4 sub-images. Left unfused these 4 Slice + Concat run on the
    full-resolution input and cost ~20%+ of end-to-end FPS (≈1000 -> ≈865 on
    YOLOX-M). The legacy ``tensorrt.deploy`` flow replaced them with a fused
    ``Focus`` plugin; we reproduce that for both FP16 and INT8/QDQ graphs.

    ORT's ``quantize_static`` emits the slices/concat with *explicit*
    QuantizeLinear/DequantizeLinear nodes, so the old name-based anchor
    (``images_DequantizeLinear_Output``) no longer matches. We walk backwards
    from ``focus_output`` (the Concat output, e.g. ``input``), deleting every
    producing node until we reach the image graph-input.
    """
    out_name = focus_output[0] if isinstance(focus_output, (list, tuple)) else focus_output
    producer = _build_producer_map(graph)
    if out_name not in producer:
        return  # nothing to fuse (already fused / unexpected layout)

    graph_inputs = {i.name for i in graph.input}
    to_delete, seen, src_tensor = [], set(), None
    quant_scale = quant_zp = None
    stack = [out_name]
    while stack:
        tensor = stack.pop()
        if tensor in graph_inputs:
            src_tensor = tensor
            continue
        node = producer.get(tensor)
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        to_delete.append(node)
        # Remember the activation quant params used inside the Focus subgraph
        # so we can re-quantize the image input the same way.
        if node.op_type == "QuantizeLinear" and quant_scale is None:
            quant_scale = node.input[1]
            quant_zp = node.input[2] if len(node.input) > 2 else None
        for inp in node.input:
            if not inp:
                continue
            if inp in graph_inputs:
                src_tensor = inp
            else:
                stack.append(inp)

    if src_tensor is None:
        return  # could not locate the image input; leave the graph untouched

    for node in to_delete:
        graph.node.remove(node)

    focus_input = src_tensor
    new_nodes = []
    if quant_scale is not None:
        # Re-create the input QDQ pair so the Focus plugin is fed a quantized
        # tensor (matches the legacy deploy flow; preserves INT8 accuracy).
        q_out = f"{src_tensor}_QuantizeLinear_Output"
        dq_out = f"{src_tensor}_DequantizeLinear_Output"
        q_in = [src_tensor, quant_scale] + ([quant_zp] if quant_zp else [])
        dq_in = [q_out, quant_scale] + ([quant_zp] if quant_zp else [])
        new_nodes.append(helper.make_node(
            "QuantizeLinear", q_in, [q_out], name=f"{src_tensor}_QuantizeLinear"))
        new_nodes.append(helper.make_node(
            "DequantizeLinear", dq_in, [dq_out], name=f"{src_tensor}_DequantizeLinear"))
        focus_input = dq_out

    new_nodes.append(helper.make_node(
        "Focus", inputs=[focus_input], outputs=[out_name],
        domain="", name="Focus_fused",
    ))
    for offset, node in enumerate(new_nodes):
        graph.node.insert(offset, node)


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

    # Fuse the Focus slice/concat subgraph into a single IxRT Focus op.
    # Works for both FP16 and INT8/QDQ graphs by walking back from the Focus
    # output to the image input (see replace_focus). Restores the FPS lost when
    # migrating from tensorrt.deploy to ORT quantization.
    if args.focus_output is not None:
        replace_focus(graph, args.focus_output)

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
