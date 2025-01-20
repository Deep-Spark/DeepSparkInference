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


class FusionConvReformat(Fusion):
    """
    Fuse FusionPVTAttention subgraph into one Attention node.
    """

    def __init__(
        self,
        model: OnnxModel,
    ):
        super().__init__(
            model,
            "FuseConvReformat_IxRT",
            ["Transpose"],
        )



    def create_fuse_node(
        self, inputs: str, outputs: str, before_conv: int, shape_data: list, prefix
    ) -> Union[NodeProto, None]:
        """Create an Attention node.

        Args:
            input (str): input name
            output (str): output name

        Returns:
            Union[NodeProto, None]: the node created or None if failed.
        """

        node_name = self.model.create_node_name(f"FuseConvReformat_{prefix}")
        node = helper.make_node(
            "FuseConvReformat_IxRT",
            inputs=inputs,
            outputs=outputs,
            name=node_name,
        )
        node.domain = "com.iluvatar"

        node.attribute.extend([helper.make_attribute("before_conv", before_conv)])
        node.attribute.extend([helper.make_attribute("shape_data", shape_data)])
        node.attribute.extend([helper.make_attribute("plugin_namespace", "")])
        node.attribute.extend([helper.make_attribute("plugin_version", "1")])
        return node

    def fuse(self, node, input_name_to_nodes, output_name_to_node):

        """
        eliminate  Transpose(linear->nchw) + Transpose 
        path: 
        ----->Transpose ---->Reshape---> conv ----->Reshape ---->Transpose--->
        
        to:
        ----->FuseConvReformat_IxRT---> conv ----->FuseConvReformat_IxRT--->
        
        """        
        start_node = node
        paths = {
            "path": (["Reshape", "Conv", "Reshape","Transpose"], [0, 0, 0, 0]),  # cross attention qery pass
        }

        nodes, path = self.match_parent_path_from_dict(start_node, paths)
        
        if nodes is None:
            logger.debug("FuseConvReformat: failed to match  path")
            return
        
        (reshape_after_node, conv_node, reshape_before_node, tranpose_before_node) = nodes

        perm1 = tranpose_before_node.attribute[0].ints
        if perm1 !=[0, 2, 1]:
            return
        perm2 = start_node.attribute[0].ints
        if perm2 !=[0, 2, 1]:
            return
        
        before_shape_data  =  numpy_helper.to_array(self.model.get_initializer(reshape_before_node.input[1]))
        
        if before_shape_data.shape[0] != 4:
            return
        
        after_shape_data  =  numpy_helper.to_array(self.model.get_initializer(reshape_after_node.input[1]))
        if after_shape_data.shape[0] != 3:
            return
        node1_inputs = tranpose_before_node.input
        node1_outputs = reshape_before_node.output
        node1_before_conv = 1
        
        new_node1 = self.create_fuse_node(
            node1_inputs, node1_outputs, node1_before_conv, before_shape_data,"before")
        
        
        node2_inputs = conv_node.output
        node2_outputs = start_node.output
        node2_before_conv = 0
        new_node2 = self.create_fuse_node(
            node2_inputs, node2_outputs, node2_before_conv, after_shape_data,"after")
        
        self.nodes_to_add.append(new_node1)
        self.nodes_to_add.append(new_node2)
        self.node_name_to_graph_name[new_node1.name] = self.this_graph_name
        self.node_name_to_graph_name[new_node2.name] = self.this_graph_name        
        self.nodes_to_remove.extend([start_node, reshape_after_node,reshape_before_node,tranpose_before_node])

