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

# !/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from tensorrt.deploy.api import GraphTransform, create_source, create_target

class Transform:
    def __init__(self, graph):
        self.t = GraphTransform(graph)
        self.graph = graph

    def ReplaceFocus(self, input_edge, outputs, to_op):
        input_var = self.graph.get_variable(input_edge)
        op = self.graph.get_operator(to_op)
        self.t.delete_operators_between_var_op(
            from_var=input_var, to_op=op
        )
        self.t.make_operator(
            "Focus", inputs=input_edge, outputs=outputs
        )
        return self.graph

    def AddYoloDecoderOp(self, inputs: list, outputs: list, op_type, **attributes):
        if attributes["anchor"] is None:
            del attributes["anchor"]
        self.t.make_operator(
            op_type, inputs=inputs, outputs=outputs, **attributes
        )
        return self.graph

    def AddConcatOp(self, inputs: list, outputs, **attributes):
        self.t.make_operator(
            "Concat", inputs=inputs, outputs=outputs, **attributes
        )
        return self.graph

def customize_ops(graph, args):
    t = Transform(graph)
    fuse_focus = args.focus_input is not None and args.focus_output is not None and args.focus_last_node is not None
    if fuse_focus:
        graph = t.ReplaceFocus(
            input_edge=args.focus_input,
            outputs=args.focus_output,
            to_op=args.focus_last_node
        )
    decoder_input = args.decoder_input_names
    num = len(decoder_input) // 3
    graph = t.AddYoloDecoderOp(
        inputs=decoder_input[:num],
        outputs=["decoder_8"],
        op_type=args.decoder_type,
        anchor=args.decoder8_anchor,
        num_class=args.num_class,
        stride=8,
        faster_impl=args.faster
    )
    graph = t.AddYoloDecoderOp(
        inputs=decoder_input[num:num*2],
        outputs=["decoder_16"],
        op_type=args.decoder_type,
        anchor=args.decoder16_anchor,
        num_class=args.num_class,
        stride=16,
        faster_impl=args.faster
    )
    graph = t.AddYoloDecoderOp(
        inputs=decoder_input[num*2:],
        outputs=["decoder_32"],
        op_type=args.decoder_type,
        anchor=args.decoder32_anchor,
        num_class=args.num_class,
        stride=32,
        faster_impl=args.faster
    )
    graph = t.AddConcatOp(
        inputs=["decoder_32", "decoder_16", "decoder_8"],
        outputs=["output"],
        axis=1
    )

    graph.outputs.clear()
    graph.add_output("output")
    graph.outputs["output"].dtype = "FLOAT"
    return graph

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str)
    parser.add_argument("--dst", type=str)
    parser.add_argument("--decoder_type", type=str, choices=["YoloV3Decoder", "YoloV5Decoder", "YoloV7Decoder", "YoloxDecoder"])
    parser.add_argument("--decoder_input_names", nargs='+', type=str)
    parser.add_argument("--decoder8_anchor", nargs='*', type=int)
    parser.add_argument("--decoder16_anchor", nargs='*', type=int)
    parser.add_argument("--decoder32_anchor", nargs='*', type=int)
    parser.add_argument("--num_class", type=int, default=80)
    parser.add_argument("--faster", type=int, default=1)
    parser.add_argument("--focus_input", type=str, default=None)
    parser.add_argument("--focus_output", type=str, default=None)
    parser.add_argument("--focus_last_node", type=str, default=None)
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()
    graph = create_source(args.src)()
    graph = customize_ops(graph, args)
    create_target(saved_path=args.dst).export(graph)
    print("Surged onnx lies on", args.dst)