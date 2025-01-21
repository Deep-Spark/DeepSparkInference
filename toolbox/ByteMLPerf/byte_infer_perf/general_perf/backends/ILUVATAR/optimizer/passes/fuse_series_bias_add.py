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

from logging import getLogger

import numpy as np
import onnx
from onnx import NodeProto, TensorProto, helper, numpy_helper

from .fusion_base import Fusion
from .fusion_utils import NumpyHelper
from .onnx_model import OnnxModel

logger = getLogger(__name__)


class FusionSerialBiasAdd(Fusion):
    def __init__(self, model: OnnxModel):
        super().__init__(model, "Add", "Softmax")

    def match_parent_path_from_dict(self, start_node, path_dict):
        res_path = None
        res_nodes = None
        for k, v in path_dict.items():
            res_nodes = self.model.match_parent_path(start_node, v[0], v[1])
            if res_nodes is None:
                continue
            return res_nodes, k
        return res_nodes, res_path

    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        paths = {
            "path1": (["Reshape", "Add", "Reshape", "Add"], [0, 0, 0, 0]),
        }
        series_nodes, path_chosen = self.match_parent_path_from_dict(node, paths)
        if not series_nodes:
            return
        last_reshape, add_2nd, _, add_1st = series_nodes

        biases = [
            self.model.get_initializer(add_1st.input[1]),
            self.model.get_initializer(add_2nd.input[1]),
        ]
        if not all(biases):
            return

        bias_arr_1st = NumpyHelper.to_array(biases[0])
        bias_arr_2nd = NumpyHelper.to_array(biases[1]).squeeze(0)
        try:
            relative_position_bias = bias_arr_1st + bias_arr_2nd
        except Exception as e:
            print("Two bias are unrelated:", e)
            return

        # Fuse
        add_name = self.model.create_node_name("Add", "Add")
        B = biases[0]
        B.CopyFrom(numpy_helper.from_array(relative_position_bias, B.name))

        fused_node = helper.make_node(
            "Add",
            inputs=[add_1st.input[0], B.name],
            outputs=last_reshape.output,
            name=add_name,
        )
        fused_node.domain = "com.iluvatar"
        self.node_name_to_graph_name[fused_node.name] = self.this_graph_name
        self.nodes_to_add.append(fused_node)
        self.nodes_to_remove.extend(series_nodes)
