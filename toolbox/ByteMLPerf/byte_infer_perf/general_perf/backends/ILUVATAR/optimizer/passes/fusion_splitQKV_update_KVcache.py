# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from logging import getLogger
from typing import Tuple, Union

from onnx import NodeProto, TensorProto, helper, numpy_helper

from .fusion_base import Fusion
from .fusion_utils import NumpyHelper
from .onnx_model import OnnxModel

logger = getLogger(__name__)


class FusionSplitQKVUpdateKVCache(Fusion):
    """
    Fuse FusionSplitQKVUpdateKVCache
    """

    def __init__(self, model: OnnxModel, hidden_size: int, num_heads: int):
        super().__init__(
            model, "SplitQKVUpdateKVCache_IxRT", "CustomQkvCrossToContext_IxRT"
        )

        self.hidden_size = hidden_size
        self.num_heads = num_heads

    def create_node(
        self,
        inputs: list,
        outputs: list,
    ) -> Union[NodeProto, None]:
        """Create an XSoftmax node.

        Args:
            data_input (str): data input name
            mask_input (str): max input name
            output (str): output name

        Returns:
            Union[NodeProto, None]: the node created or None if failed.
        """
        node_name = self.model.create_node_name("SplitQKVUpdateKVCache_IxRT")

        new_node = helper.make_node(
            "SplitQKVUpdateKVCache_IxRT",
            inputs=inputs,
            outputs=outputs,
            name=node_name,
        )
        new_node.domain = "com.iluvatar"
        new_node.attribute.extend([helper.make_attribute("plugin_namespace", "")])
        new_node.attribute.extend([helper.make_attribute("plugin_version", "1")])
        new_node.attribute.extend([helper.make_attribute("num_head", self.num_heads)])
        new_node.attribute.extend(
            [helper.make_attribute("head_dim", self.hidden_size // self.num_heads)]
        )

        return new_node

    def fuse(self, node, input_name_to_nodes, output_name_to_node):

        query_paths = {
            "query_path": (
                ["Transpose", "Reshape", "Split"],
                [0, 0, None],
            ),
        }

        key_paths = {
            "key_path": (
                ["Concat", "Transpose", "Reshape", "Split"],
                [1, None, 0, None],
            ),
        }

        value_paths = {
            "value_path": (
                ["Concat", "Transpose", "Reshape", "Split"],
                [2, None, 0, None],
            ),
        }

        q_nodes, q_path = self.match_parent_path_from_dict(node, query_paths)

        k_nodes, k_path = self.match_parent_path_from_dict(node, key_paths)

        v_nodes, v_path = self.match_parent_path_from_dict(node, value_paths)

        if (q_nodes is not None) and (k_nodes is not None) and (v_nodes is not None):
            (q_transpose_node, q_reshape_node, q_split_node) = q_nodes
            (k_concat_node, k_transpose_node, k_reshape_node, k_split_node) = k_nodes

            (v_concat_node, v_transpose_node, v_reshape_node, v_split_node) = v_nodes

            inputs = [
                q_split_node.input[0],
                k_concat_node.input[0],
                v_concat_node.input[0],
            ]

            outputs = [
                q_transpose_node.output[0],
                k_concat_node.output[0],
                v_concat_node.output[0],
            ]

            new_node = self.create_node(inputs, outputs)

            self.nodes_to_add.append(new_node)
            self.node_name_to_graph_name[new_node.name] = self.this_graph_name
            self.nodes_to_remove.append(q_transpose_node)
            self.nodes_to_remove.append(q_reshape_node)
            self.nodes_to_remove.append(q_split_node)

            self.nodes_to_remove.append(k_concat_node)
            self.nodes_to_remove.append(k_transpose_node)
            self.nodes_to_remove.append(k_reshape_node)

            self.nodes_to_remove.append(v_concat_node)
            self.nodes_to_remove.append(v_transpose_node)
            self.nodes_to_remove.append(v_reshape_node)

        else:
            return