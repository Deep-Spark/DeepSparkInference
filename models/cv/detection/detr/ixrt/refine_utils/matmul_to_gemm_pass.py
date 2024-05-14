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

from refine_utils.common import *

#
#   Common pattern Matmul to Gemm
#
class FusedGemmPass(BasePass):

    def process(self, graph: Graph) -> Graph:
        self.transform = GraphTransform(graph)

        find_sequence_subgraph(
            graph, pattern=[OP.MATMUL], callback=self.to_gemm, strict=True
        )
        return graph

    def to_gemm(self, graph, pattern: PatternGraph):
        matmul_op = pattern.nodes[0]
        inputs = matmul_op.operator.inputs
        outputs = matmul_op.operator.outputs

        if len(inputs)!=2 and len(outputs)!=1:
            return

        for input in inputs:
            if self.transform.is_leaf_variable(input):
                return

        print(f"{self.transform.get_variable(inputs[0]).shape}   {self.transform.get_variable(inputs[1]).shape}")
        self.transform.delete_operator(matmul_op.operator)

        op = self.transform.make_operator(
            op_type = "Gemm",
            inputs = inputs,
            outputs = outputs,
            alpha = 1,
            beta = 1,
            transB = 1
        )

        self.transform.add_operator(op)