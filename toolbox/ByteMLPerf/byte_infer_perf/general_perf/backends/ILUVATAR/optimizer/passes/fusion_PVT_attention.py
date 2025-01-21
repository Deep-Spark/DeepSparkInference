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


class FusionPVTAttention(Fusion):
    """
    Fuse FusionPVTAttention subgraph into one Attention node.
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
        self.num_heads_warning = False
        self.hidden_size_warning = False


    def create_decoder_attention_node(
        self, inputs: str, outputs: str, type_mask: int, has_mask: int,scale: float
    ) -> Union[NodeProto, None]:
        """Create an Attention node.

        Args:
            input (str): input name
            output (str): output name

        Returns:
            Union[NodeProto, None]: the node created or None if failed.
        """

        attention_node_name = self.model.create_node_name("cross_Attention")
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
        path:

         (query) ---------------->MatMul ---->Mul --->softmax --->MatMul--->
                                    /                             /
         (key)   ---->Transpose -->                              /
                                                                /
                                                               /
                                                              /
         (value)--------------------------------------------->

        """

        start_node = node
        qkv_paths = {
            "path": (["Mul", "MatMul", "Transpose"], [0, 0, 0]),  # cross attention qery pass
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
        attention_outputs = second_matmul_node.output
        remove_nodes = [second_matmul_node, node]



        (mul_node, first_matmul_node, transpose_node) = qkv_nodes
        transpose_nodes = self.model.get_parents(first_matmul_node)
        
        q_input = transpose_nodes[0].output[0]
        k_input = transpose_nodes[1].input[0]
        v_input = second_matmul_node.input[1]
        attention_inputs = [q_input, k_input, v_input]
        remove_nodes.extend([first_matmul_node, mul_node, transpose_nodes[1]])

        has_mask = 0
        type_mask = 4 
        
        scale =  numpy_helper.to_array(self.model.get_initializer(mul_node.input[1])).item()                
        atten_node = self.create_decoder_attention_node(
            attention_inputs, attention_outputs, type_mask, has_mask,scale
        )
        self.nodes_to_add.append(atten_node)
        self.node_name_to_graph_name[atten_node.name] = self.this_graph_name
        self.nodes_to_remove.extend(remove_nodes)