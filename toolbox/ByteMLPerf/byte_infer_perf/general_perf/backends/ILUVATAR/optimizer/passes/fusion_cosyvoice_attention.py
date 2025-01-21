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
import math
from enum import Enum
from logging import getLogger
from os import name
from sys import path
from typing import Tuple, Union

import numpy as np
import onnx
from onnx import NodeProto, TensorProto, helper, numpy_helper

from .fusion_base import Fusion
from .fusion_options import AttentionMaskFormat
from .fusion_utils import FusionUtils, NumpyHelper
from .onnx_model import OnnxModel
from .shape_infer_helper import SymbolicShapeInferenceHelper, get_shape_from_type_proto

logger = getLogger(__name__)



class FusionCosyvoiceAttention(Fusion):
    """
    Fuse T5Attention subgraph into one Attention node.
    """

    def __init__(
        self,
        model: OnnxModel,
    ):
        super().__init__(
            model,
            "CustomQkvCrossToContext_IxRT",
            ["Softmax"],
        )

        # Flags to show warning only once
        self.num_heads_warning = True
        self.hidden_size_warning = True

    def get_num_heads_and_hidden_size(self, reshape_q: NodeProto) -> Tuple[int, int]:
        """Detect num_heads and hidden_size from a reshape node.

        Args:
            reshape_q (NodeProto): reshape node for Q

        Returns:
            Tuple[int, int]: num_heads and hidden_size
        """

        # we assume that reshape fusion has done, so the shape is a tensor like [0, 0, num_heads, head_size]
        q_shape = self.model.get_initializer(reshape_q.input[1])
        if q_shape is None:
            logger.debug(f"{reshape_q.input[1]} is not initializer.")
            return [0, 0]

        q_shape_value = NumpyHelper.to_array(q_shape)
        if len(q_shape_value) != 4 or (q_shape_value[2] <= 0 or q_shape_value[3] <= 0):
            logger.debug(
                f"q_shape_value={q_shape_value}. Expected value are like [0, 0, num_heads, head_size]."
            )
            return [0, 0]

        num_heads = q_shape_value[2]
        head_size = q_shape_value[3]
        hidden_size = num_heads * head_size

        return num_heads, hidden_size

    def create_decoder_attention_node(
        self, inputs: str, outputs: str, type_mask: int, has_mask: int, scale: float
    ) -> Union[NodeProto, None]:
        """Create an Attention node.

        Args:
            input (str): input name
            output (str): output name

        Returns:
            Union[NodeProto, None]: the node created or None if failed.
        """

        attention_node_name = self.model.create_node_name("decoder_Attention")
        attention_node = helper.make_node(
            "CustomQkvCrossToContext_IxRT",
            inputs=inputs,
            outputs=outputs,
            name=attention_node_name,
        )
        attention_node.domain = "com.iluvatar"
        attention_node.attribute.extend([helper.make_attribute("type_id", 2)])
        attention_node.attribute.extend([helper.make_attribute("scale", scale)])
        attention_node.attribute.extend([helper.make_attribute("has_mask", has_mask)])
        attention_node.attribute.extend([helper.make_attribute("plugin_namespace", "")])
        attention_node.attribute.extend([helper.make_attribute("plugin_version", "1")])
        attention_node.attribute.extend([helper.make_attribute("type_mask", type_mask)])

        return attention_node

    def fuse(self, node, input_name_to_nodes, output_name_to_node):

        """
         path1:

         (query) --------------MatMul---Div --> add -->softmax --->MatMul--->
                                 /             /                    /
         (key)   ---->Transpose >             /                    /
                                             /                    /
         (mask)   ------------------------>                     /
                                                               /
         (value)--------------------------------------------->
         """




        import pdb
        start_node = node
        qkv_paths = {
            "path1": (
                ["Add", "Div", "MatMul", "Transpose"],
                [None, 0, None, 1],
            ),  # float mask self attention,self attention key pass
        }

        qkv_nodes, qkv_path = self.match_parent_path_from_dict(start_node, qkv_paths)
        
        if qkv_nodes is None:
            logger.debug("fuse_attention: failed to match qkv path")
            return
        next_nodes = self.model.get_children(node)
    
        if len(next_nodes) == 0:
            return
                
        if next_nodes[0].op_type != "MatMul":
            return
        
        second_matmul_node = next_nodes[0]
        attention_inputs = None
        attention_outputs = second_matmul_node.output
        remove_nodes = [second_matmul_node, node]

        (add_node, div_node, first_matmul_node, transpose_node) = qkv_nodes
        transpose_nodes = self.model.get_parents(first_matmul_node)
        q_input = transpose_nodes[0].output[0]
        
        k_transpose_node = transpose_nodes[1]
        k_transpose_node_perm = k_transpose_node.attribute[0].ints
        
        if  k_transpose_node_perm == [0, 2, 3, 1]:  #transpose has bean merge,[0,2,1,3]->[0, 1, 3, 2] = [0, 2, 3, 1]
            k_input = transpose_nodes[1].output[0]
            
            transpose_nodes[1].attribute[0].ints[0] = 0 
            transpose_nodes[1].attribute[0].ints[1] = 2 
            transpose_nodes[1].attribute[0].ints[2] = 1 
            transpose_nodes[1].attribute[0].ints[3] = 3 
            
            remove_nodes.extend([add_node, div_node, first_matmul_node])
            
        elif k_transpose_node_perm == [0, 1, 3, 2]:
            k_input = transpose_nodes[1].input[0]
            remove_nodes.extend([add_node, div_node, first_matmul_node,k_transpose_node])
            
        else:
            return         
        
        v_input = second_matmul_node.input[1]
        attention_inputs = [q_input, k_input, v_input]
        
        has_mask = 1
        type_mask = 3 # float mask
        
        mask_input = add_node.input[0]
        score_out = div_node.output[0]
        if add_node.input[0] == score_out:
            mask_input = add_node.input[1]
        attention_inputs.append(mask_input)
        
        scale_data = self.model.get_initializer_input_edges(div_node.name, return_np_array = True)
        scale = 1.0 / scale_data[0]
        
        atten_node = self.create_decoder_attention_node(
            attention_inputs, attention_outputs, type_mask, has_mask, scale
        )
        
        self.nodes_to_add.append(atten_node)
        self.node_name_to_graph_name[atten_node.name] = self.this_graph_name
        self.nodes_to_remove.extend(remove_nodes)
        
