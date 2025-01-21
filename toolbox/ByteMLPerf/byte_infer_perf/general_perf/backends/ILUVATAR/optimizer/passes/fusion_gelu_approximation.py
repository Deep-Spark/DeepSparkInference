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
#

# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from logging import getLogger

from onnx import helper

from .fusion_base import Fusion
from .onnx_model import OnnxModel


class FusionGeluApproximation(Fusion):
    def __init__(self, model: OnnxModel):
        super().__init__(model, "FastGelu", ["Gelu", "BiasGelu"], "GeluApproximation")

    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        new_node = helper.make_node(
            "FastGelu",
            inputs=node.input,
            outputs=node.output,
            name=self.model.create_node_name(
                "FastGelu", node.op_type + "_Approximation"
            ),
        )
        new_node.domain = "com.microsoft"
        self.nodes_to_remove.append(node)
        self.nodes_to_add.append(new_node)
        self.node_name_to_graph_name[new_node.name] = self.this_graph_name
