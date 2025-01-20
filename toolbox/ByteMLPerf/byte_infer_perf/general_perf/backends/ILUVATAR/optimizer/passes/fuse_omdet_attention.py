
from logging import getLogger
from typing import Dict

import math
import numpy as np
from onnx import TensorProto, helper

from .fusion_base import Fusion
from .onnx_model import OnnxModel

logger = getLogger(__name__)

class FusionLayerOmdetAttention(Fusion):
    def __init__(self, model: OnnxModel):
        super().__init__(
            model, "CustomQKVToContextPluginDynamic_IxRT", "CustomFCPluginDynamic_IxRT"
        )

    def fuse(self, node, input_name_to_nodes: Dict, output_name_to_node: Dict):
        """
        [Root] -->  CustomFCPluginDynamic_IxRT-->  CustomQKVToContextPluginDynamic_IxRT  --> CustomFCPluginDynamic_IxRT
        """
        children = self.model.get_children(node, input_name_to_nodes)
        parent = self.model.get_parents(node, output_name_to_node)
        
        if len(children) != 1:
            return
        if len(parent) != 1:
            return

        fc_first_node = None
        for par in parent:
            fc_first_node = self.model.find_first_parent_by_type(
                par, "CustomFCPluginDynamic_IxRT", output_name_to_node, recursive=True
            )
            if fc_first_node is not None:
                break
        if fc_first_node is None:
            return
        
        start_node = node
        
        # v path
        v_nodes = self.model.match_parent_path(
            start_node,
            ["Reshape", "Transpose", "MatMul", "Gather", "Transpose", "Reshape"],
            [0, 0, 0, 1, 0, 0],
            output_name_to_node,
        )
        
        # path1, q and k path
        q_nodes = self.model.match_parent_path(
            start_node,
            ["Reshape", "Transpose", "MatMul", "Softmax", "Add", "MatMul", "Transpose", "Gather", "Transpose", "Reshape"],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            output_name_to_node,
        )
        
        k_nodes = self.model.match_parent_path(
            start_node,
            ["Reshape", "Transpose", "MatMul", "Softmax", "Add", "MatMul", "Mul", "Gather", "Transpose", "Reshape"],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            output_name_to_node,
        )
    
        # path2, q and k path
        q_nodes_1 = self.model.match_parent_path(
            start_node,
            ["Reshape", "Transpose", "MatMul", "Softmax", "Reshape", "Add", "Reshape", "Add", "MatMul", "Transpose", "Gather", "Transpose", "Reshape"],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            output_name_to_node,
        )
        
        k_nodes_1 = self.model.match_parent_path(
            start_node,
            ["Reshape", "Transpose", "MatMul", "Softmax", "Reshape", "Add", "Reshape", "Add", "MatMul", "Mul", "Gather", "Transpose", "Reshape"],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            output_name_to_node,
        )
        
        if v_nodes is None:
            return
        
        if v_nodes and q_nodes and k_nodes:
            subgraph_nodes = []
            subgraph_nodes.extend(q_nodes)
            subgraph_nodes.extend(k_nodes)
            subgraph_nodes.extend(v_nodes)
            
            subgraph_nodes_unique = []
            for item in subgraph_nodes:
                if item not in subgraph_nodes_unique:
                    subgraph_nodes_unique.append(item)
            
            add_node = q_nodes[4]
            hidden_size = start_node.attribute[0].i
            _, mul_val = self.model.get_constant_input(k_nodes[6])
            num_heads = hidden_size // math.floor((1/mul_val)*(1/ mul_val))
            attention_input_1_name = add_node.input[1]
        
        if v_nodes and q_nodes_1 and k_nodes_1:
            subgraph_nodes = []
            subgraph_nodes.extend(q_nodes_1)
            subgraph_nodes.extend(k_nodes_1)
            subgraph_nodes.extend(v_nodes)
            
            subgraph_nodes_unique = []
            for item in subgraph_nodes:
                if item not in subgraph_nodes_unique:
                    subgraph_nodes_unique.append(item)
            
            hidden_size = start_node.attribute[0].i
            _, mul_val = self.model.get_constant_input(k_nodes_1[9])
            num_heads = hidden_size // math.floor((1/mul_val)*(1/ mul_val))
            
            add_1 = self.model.get_initializer(q_nodes_1[5].input[1], True)
            add_2 = self.model.get_initializer(q_nodes_1[7].input[1], True)
            add_all = np.squeeze(add_1 + add_2)
            
            attention_input_1_name = "attention_" + q_nodes_1[5].input[1]
            attention_input_1 = helper.make_tensor(
                attention_input_1_name, TensorProto.FLOAT, add_all.shape, add_all.flatten().tolist())
            
            self.model.add_initializer(attention_input_1, self.this_graph_name)
            
        attention_node = helper.make_node(
            "CustomQKVToContextPluginDynamic_IxRT",
            inputs=[fc_first_node.output[0], attention_input_1_name],
            outputs=[start_node.input[0]],
            name=self.model.create_node_name(
                "OmdetAttention", name_prefix="OmdetAttention"
            ),
        )
        attention_node.domain = "com.iluvatar"
        attention_node.attribute.extend([helper.make_attribute("type_id", 2)])
        attention_node.attribute.extend([helper.make_attribute("num_heads", num_heads)])
        attention_node.attribute.extend([helper.make_attribute("hidden_size", hidden_size)])
        attention_node.attribute.extend([helper.make_attribute("has_mask", 1)])
        attention_node.attribute.extend([helper.make_attribute("plugin_namespace", "")])
        attention_node.attribute.extend([helper.make_attribute("plugin_version", "1")])
        attention_node.attribute.extend([helper.make_attribute("has_qk_bias", 1)])
        
        self.nodes_to_remove.extend(subgraph_nodes_unique)
        
        self.nodes_to_add.append(attention_node)
        self.node_name_to_graph_name[attention_node.name] = self.this_graph_name
        
        