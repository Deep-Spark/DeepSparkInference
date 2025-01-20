
from logging import getLogger
from typing import Dict

import numpy as np
from onnx import TensorProto, helper

from .fusion_base import Fusion
from .onnx_model import OnnxModel

logger = getLogger(__name__)

class FusionLayerInverseSigmoid(Fusion):
    def __init__(self, model: OnnxModel):
        super().__init__(
            model, "InverseSigmoid", "Clip"
        )

    def fuse(self, node, input_name_to_nodes: Dict, output_name_to_node: Dict):
        """
                     +------------Clip-----------+
                     |                           |
                     |                           v
        [Root] -->  Clip-->  Sub  --> Clip --> Div --> Log
        """
        children = self.model.get_children(node, input_name_to_nodes)
        if len(children) != 2:
            return

        root_input = node.input[0]

        if not ((children[0].op_type == "Sub" and children[1].op_type == "Clip") or (children[0].op_type == "Clip" and children[1].op_type == "Sub")):
            return

        log_node = None
        for child in children:
            log_node = self.model.find_first_child_by_type(
                child, "Log", input_name_to_nodes, recursive=True
            )
            if log_node is not None:
                break
        if log_node is None:
            return
        parent_nodes = self.model.match_parent_path(
            log_node,
            ["Div", "Clip", "Sub", "Clip"],
            [0, 1, 0, 1],
            output_name_to_node,
        )
        if parent_nodes is None:
            return

        sub_node = parent_nodes[2]
        if sub_node not in children:
            return

        div_node = parent_nodes[0]
        div_parents_nodes = self.model.get_parents(div_node)
        if len(div_parents_nodes) != 2:
            return
        if div_parents_nodes[0].op_type != "Clip":
            return
        if div_parents_nodes[0] not in children:
            return

        subgraph_nodes = [node]
        subgraph_nodes.extend([log_node])
        subgraph_nodes.extend(parent_nodes)
        subgraph_nodes.extend([div_parents_nodes[0]])
        _, eps_val = self.model.get_constant_input(div_parents_nodes[0])

        self.nodes_to_remove.extend(subgraph_nodes)
        inverse_sigmoid_node = helper.make_node(
            "InverseSigmoid",
            inputs=[node.input[0]],
            outputs=[log_node.output[0]],
            name=self.model.create_node_name(
                "InverseSigmoid", name_prefix="InverseSigmoid"
            ),
        )
        inverse_sigmoid_node.attribute.extend(
            [helper.make_attribute("epsilon", float(eps_val))]
        )
        self.nodes_to_add.append(inverse_sigmoid_node)
        self.node_name_to_graph_name[inverse_sigmoid_node.name] = self.this_graph_name