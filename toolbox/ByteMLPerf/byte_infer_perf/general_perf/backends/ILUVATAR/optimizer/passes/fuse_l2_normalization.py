from logging import getLogger
from typing import Dict

import numpy as np
from onnx import TensorProto, helper

from .fusion_base import Fusion
from .onnx_model import OnnxModel

logger = getLogger(__name__)

class FusionLayerL2Normalization(Fusion):
    def __init__(self, model: OnnxModel):
        super().__init__(
            model, "L2Normalization", "Abs"
        )

    def fuse(self, node, input_name_to_nodes: Dict, output_name_to_node: Dict):
        """
            +-------------------------------------------------------+
            |                                                       |
            |                                                       v
        [Root] -->  Abs-->  Pow  --> ReduceSum --> Pow --> Clip --> Div
        """
        pow1_nodes = self.model.get_children(node, input_name_to_nodes)
        if len(pow1_nodes) != 1 or pow1_nodes[0].op_type != "Pow":
            return

        reduce_nodes = self.model.get_children(pow1_nodes[0], input_name_to_nodes)
        if len(reduce_nodes) != 1 or reduce_nodes[0].op_type != "ReduceSum":
            return

        pow2_nodes = self.model.get_children(reduce_nodes[0], input_name_to_nodes)
        if len(pow2_nodes) != 1 or pow2_nodes[0].op_type != "Pow":
            return

        clip_nodes = self.model.get_children(pow2_nodes[0], input_name_to_nodes)
        if len(clip_nodes) != 1 or clip_nodes[0].op_type != "Clip":
            return

        div_nodes = self.model.get_children(clip_nodes[0], input_name_to_nodes)
        if len(div_nodes) != 1 or div_nodes[0].op_type != "Div":
            return

        root_input = node.input[0]
        if div_nodes[0].input[0] != root_input:
            return

        subgraph_nodes = [node, pow1_nodes[0], reduce_nodes[0], pow2_nodes[0], clip_nodes[0], div_nodes[0]]
        _, eps_val = self.model.get_constant_input(clip_nodes[0])
        _, norm_axes = self.model.get_constant_input(reduce_nodes[0])
        norm_axes = norm_axes.astype(np.int32)

        self.nodes_to_remove.extend(subgraph_nodes)
        l2_normalization_node = helper.make_node(
            "L2Normalization",
            inputs=[node.input[0]],
            outputs=[div_nodes[0].output[0]],
            name=self.model.create_node_name(
                "L2Normalization", name_prefix="L2Normalization"
            ),
        )
        l2_normalization_node.attribute.extend(
            [helper.make_attribute("epsilon", float(eps_val)), 
             helper.make_attribute("axes", norm_axes),
             helper.make_attribute("axes_length", int(norm_axes.size))]
        )
        self.nodes_to_add.append(l2_normalization_node)
        self.node_name_to_graph_name[l2_normalization_node.name] = self.this_graph_name