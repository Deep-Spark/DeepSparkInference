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
import dataclasses

from refine_utils.common import *

# AXB=C, Only for B is initializer

class FusedLinearPass(BasePass):

    def process(self, graph: Graph) -> Graph:
        self.transform = GraphTransform(graph)

        find_sequence_subgraph(
            graph, pattern=[OP.MATMUL, OP.ADD], callback=self.to_linear_with_bias, strict=True
        )
        find_sequence_subgraph(
            graph, pattern=[OP.MATMUL], callback=self.to_linear, strict=True
        )
        return graph

    def to_linear_with_bias(self, graph, pattern: PatternGraph):
        matmul = pattern.nodes[0]
        add = pattern.nodes[1]
        if len(add.operator.inputs) != 2:
            return

        b_var = graph.get_variable(matmul.operator.inputs[1])
        if not graph.is_leaf_variable(b_var) or b_var.value is None:
            return

        if b_var.value.ndim != 2:
            return

        bias_var = None
        for input in add.operator.inputs:
            if input not in matmul.operator.outputs:
                bias_var = input

        inputs = matmul.operator.inputs
        inputs.append(bias_var)
        outputs = add.operator.outputs

        b_var.value =  b_var.value.transpose(1, 0)
        b_var.shape[0],b_var.shape[1] = b_var.shape[1],b_var.shape[0]
        
        hidden_size = b_var.shape[1]
        linear_dim = b_var.shape[0]
        
        attributes = {
            "hidden_size": hidden_size,
            "linear_dim":  linear_dim,
            "has_bias": 1,
            "act_type":"none"
        }
        
        self.transform.make_operator(
            "LinearFP16",
            inputs=inputs,
            outputs=outputs,
            **attributes
        )
        
        self.transform.delete_operator(add.operator)
        self.transform.delete_operator(matmul.operator)

    def to_linear(self, graph, pattern: PatternGraph):
        matmul = pattern.nodes[0]
        if len(matmul.operator.inputs) != 2:
            return

        b_var = graph.get_variable(matmul.operator.inputs[1])
        if not graph.is_leaf_variable(b_var) or b_var.value is None:
            return

        if b_var.value.ndim != 2:
            return

        attributes = {
            "hidden_size": hidden_size,
            "linear_dim":  linear_dim,
            "has_bias":    0,
            "act_type":    "none"
        }

        b_var.value =  b_var.value.transpose(1, 0)
        b_var.shape[0],b_var.shape[1] = b_var.shape[1], b_var.shape[0]
        
        hidden_size = b_var.shape[1]
        linear_dim = b_var.shape[0]

        op = self.transform.make_operator(
            op_type = "LinearFP16",
            inputs = pattern.nodes[0].operator.inputs,
            outputs=[pattern.nodes[-1].operator.outputs[0]],
            **attributes
        )

        self.transform.add_operator(op)

        self.transform.delete_operator(matmul.operator)