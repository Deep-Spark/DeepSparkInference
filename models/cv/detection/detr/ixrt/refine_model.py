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

import os
import argparse
import dataclasses

import torch
import onnx

from refine_utils.matmul_to_gemm_pass import FusedGemmPass
from refine_utils.linear_pass import FusedLinearPass

from refine_utils.common import *

def get_constant_input_name_of_operator(graph: Graph, operator: Operator):
    const = None
    for input in operator.inputs:
        if not graph.containe_var(input):
            continue

        if not graph.is_leaf_variable(input):
            continue

        input_var = graph.get_variable(input)
        if input_var.value is not None:
            const = input
    return const 

class FuseLayerNormPass(BasePass):

    def process(self, graph: Graph) -> Graph:
        self.transform = GraphTransform(graph)
        find_sequence_subgraph(
            graph,
            [OP.REDUCE_MEAN, OP.SUB, OP.POW, OP.REDUCE_MEAN, OP.ADD, OP.SQRT, OP.DIV, OP.MUL, OP.ADD],
            self.fuse_layer_norm,
            strict=False
        )
        return graph

    def fuse_layer_norm(self, graph: Graph, pattern: PatternGraph):
        # 检查 REDUCE_MEAN 的输入是否和 SUB 的输入是一致的
        if pattern.nodes[0].operator.inputs[0] != pattern.nodes[1].operator.inputs[0]:
            return

        # 检查 POW 的输入是否和 DIV 的输入是一致的
        if pattern.nodes[2].operator.inputs[0] != pattern.nodes[6].operator.inputs[0]:
            return

        # 检查部分算子的输出是否被多个算子使用
        nodes = pattern.nodes
        for node in [nodes[0]] + nodes[2:-1]:
            next_ops = graph.get_next_operators(node.operator)
            if len(next_ops) > 1:
                return

        eps = None
        for input in nodes[4].operator.inputs:
            input_var = graph.get_variable(input)
            if input_var.value is not None and graph.is_leaf_variable(input):
                eps = to_py_type(input_var.value)

        scale = get_constant_input_name_of_operator(graph, nodes[-2].operator)
        bias = get_constant_input_name_of_operator(graph, nodes[-1].operator)

        self.transform.delete_operators_between_op_op(nodes[0].operator, nodes[-1].operator)
        
        bias_var = graph.get_variable(bias)
        print(bias_var)
        
        attributes = {
            "axis": nodes[0].operator.attributes.axes,
            "epsilon": eps,
        }
        
        
        layer_norm_op = self.transform.make_operator(
            op_type="LayerNormalization",
            inputs=[nodes[0].operator.inputs[0], scale, bias],
            outputs=[nodes[-1].operator.outputs[0]],
            **attributes
        )

        self.transform.add_operator(layer_norm_op)

class FusedGeluPass(BasePass):

    def process(self, graph: Graph) -> Graph:
        self.transform = GraphTransform(graph)

        find_sequence_subgraph(
            graph, pattern=[OP.DIV, OP.ERF, OP.ADD, OP.MUL, OP.MUL], callback=self.fuse_gelu, strict=True
        )
        return graph

    def fuse_gelu(self, graph: Graph, pattern: PatternGraph):
        nodes = pattern.nodes
        prev_op = self.transform.get_previous_operators(nodes[0].operator)[0]
        next_ops = self.transform.get_next_operators(prev_op)
        if len(next_ops) != 2:
            return

        if nodes[0].operator not in next_ops or nodes[3].operator not in next_ops:
            return

        gelu_op_input = None
        for input in nodes[3].operator.inputs:
            if input in nodes[0].operator.inputs:
                gelu_op_input = input
                break

        self.transform.delete_operators_between_op_op(nodes[0].operator, nodes[-1].operator)

        gelu_op = self.transform.make_operator(
            op_type=OP.GELU,
            inputs=[gelu_op_input],
            outputs=[nodes[-1].operator.outputs[0]]
        )
        self.transform.add_operator(gelu_op)

@dataclasses.dataclass
class NormalizeAttr(BaseOperatorAttr):
    p: float = 2.0
    epsilon: float = 1e-12
    axis: int = 1


@registe_operator(OP.GELU)
class GeluOperator(BaseOperator):

    def call(
        self,
        executor,
        operator: Operator,
        inputs: List,
        attr: NormalizeAttr,
    ):
        return F.gelu(inputs[0])

    def convert_onnx_operator(
        self, ir_graph: Graph, onnx_graph: onnx.GraphProto, node: onnx.NodeProto
    ) -> Operator:
        return default_converter(ir_graph, onnx_graph, node, attr_cls=attr.EmptyAttr)

    def quantize(
        self,
        graph: Graph,
        op: Operator,
        operator_observer_config: QuantOperatorObserverConfig,
        quant_outputs: bool = False,
    ):
        return quant_single_input_operator(graph, op, operator_observer_config, quant_outputs=quant_outputs)



class ClearUnsedVariables(BasePass):

    def process(self, graph: Graph) -> Graph:
        vars = list(graph.variables)

        for var in vars:
            if len(graph.get_dst_operators(var)) == 0 and graph.is_leaf_variable(var):
                graph.delete_variable(var)

        quant_params = list(graph.quant_parameters.keys())
        for var in quant_params:
            if not graph.containe_var(var):
                graph.quant_parameters.pop(var)

        return graph

class FormatLayerNorm(BasePass):

    def process(self, graph: Graph) -> Graph:
        for op in graph.operators.values():
            if "LayerNormalization" in op.op_type:
                self.format_layer_norm(graph, op)
        return graph

    def format_layer_norm(self, graph, operator):
        if not hasattr(operator.attributes, "axis"):
            return
        if isinstance(operator.attributes.axis, (tuple, list)):
            operator.attributes.axis = operator.attributes.axis[0]

class FormatReshape(BasePass):

    def process(self, graph: Graph) -> Graph:
        for op in graph.operators.values():
            if op.op_type == "Reshape":
                self.format_reshape(graph, op)

        return graph

    def format_reshape(self, graph, operator):
        shape = graph.get_variable(operator.inputs[1])
        shape.value = torch.tensor(shape.value, dtype=torch.int64)

class FormatScalar(BasePass):

    def process(self, graph: Graph):
        for var in graph.variables.values():
            var: Variable
            use_ops = graph.get_dst_operators(var)

            if len(use_ops) == 0:
                continue

            if use_ops[0].op_type not in [OP.MUL, OP.ADD, OP.GATHER]:
                continue

            if var.value is not None and var.value.ndim == 0:
                var.value = var.value.reshape(1)
                print(f"Reshape scalar to tensor for {var.name}.")

        return graph

class RenamePass(BasePass):

    def process(self, graph:Graph):

        names = [name for name in graph.operators.keys()]
        for old_name in names:
            new_name = old_name.replace("/", "#")

            graph.rename_operator(old_name, new_name)

        names = [name for name in graph.variables.keys()]
        for name in names:
            new_name = name.replace("/", ".").replace("Output", "out").replace("output", "out")

            graph.rename_vaiable(name, new_name,
                                with_variables=True, 
                                with_operator_outputs=True)

        return graph

def create_pipeline(example_inputs):
    return PassSequence(
        FuseLayerNormPass(),
        FusedGeluPass(),

        ClearUnsedVariables(),
        FormatLayerNorm(),
        FormatReshape(),
        # FormatScalar(),
        # RenamePass()
    )

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_path", type=str)
    parser.add_argument("--dst_onnx_path", type=str)

    parser.add_argument("--bsz", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--imgsz", type=int, default=224,
                        help="Image size")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    example_inputs = torch.randn(args.bsz, 3, args.imgsz, args.imgsz)

    refine_pipline = Pipeline(
        create_source(f"{args.onnx_path}", example_inputs=example_inputs),
        create_pipeline(example_inputs),
        create_target(
            f"{args.dst_onnx_path}",
            example_inputs=example_inputs,
        )
    )
    refine_pipline.run()

    print(f"refine the model, input shape={example_inputs.shape}")
