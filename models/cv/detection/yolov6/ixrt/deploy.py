# !/usr/bin/env python
# -*- coding: utf-8 -*-
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
import argparse
import copy

from typing import Union, Callable, List

from tensorrt.deploy.api import *
from tensorrt.deploy.backend.onnx.converter import default_converter
from tensorrt.deploy.backend.torch.executor.operators._operators import to_py_type
from tensorrt.deploy.ir.operator_attr import BaseOperatorAttr, EmptyAttr
from tensorrt.deploy.ir.operator_type import OperatorType as OP
from tensorrt.deploy.ir import operator_attr as attr, Operator, generate_operator_name
from tensorrt.deploy.fusion import BasePass, PatternGraph, build_sequence_graph, GraphMatcher, PassSequence
from tensorrt.deploy.ir import Graph
from tensorrt.deploy.quantizer.quant_operator.base import quant_single_input_operator
from tensorrt.deploy.backend.onnx.converter import convert_onnx_operator
from tensorrt.deploy.api import GraphTransform, create_source, create_target

class FuseSiLUPass(BasePass):
    def process(self, graph: Graph) -> Graph:
        pattern = build_sequence_graph([OP.SIGMOID, OP.MUL])

        matcher = GraphMatcher(pattern, strict=False)
        self.transform = GraphTransform(graph)
        matcher.findall(graph, self.fuse_mish)
        return graph

    def fuse_mish(self, graph: Graph, pattern_graph: PatternGraph):
        sigmoid = pattern_graph.nodes[0].operator
        mul = pattern_graph.nodes[-1].operator

        if not self.can_fused(graph, pattern_graph):
            return

        self.transform.delete_operators_between_op_op(sigmoid, mul)

        silu_op = Operator(
            name=generate_operator_name(graph, pattern="SiLU_{idx}"),
            op_type=OP.SILU,
            inputs=copy.copy(sigmoid.inputs),
            outputs=copy.copy(mul.outputs),
        )
        silu_op.is_quant_operator = sigmoid.is_quant_operator and mul.is_quant_operator
        graph.add_operator(silu_op)

    def can_fused(self, graph: Graph, pattern_graph: PatternGraph):
        sigmoid = pattern_graph.nodes[0].operator
        mul = pattern_graph.nodes[-1].operator

        # 如果 sigmoid 的结果 被多个 OP 使用，则不能融合
        if len(self.transform.get_next_operators(sigmoid)) > 1:
            return False

        # 检查 mul 的输入是不是和 sigmoid 是同源的
        softplus_prev_op = graph.get_previous_operators(sigmoid)
        if len(softplus_prev_op) != 1:
            return False

        mul_prev_op = graph.get_previous_operators(mul)
        if len(mul_prev_op) != 2:
            return False

        for op in mul_prev_op:
            if op is softplus_prev_op[0]:
                return True

        return False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str)
    parser.add_argument("--dst", type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()
    graph = create_source(args.src)()
    graph = FuseSiLUPass().process(graph)
    create_target(saved_path=args.dst).export(graph)
    print("Surged onnx lies on", args.dst)
