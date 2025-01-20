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


class FusionCosyVoiceSplitQKVUpdateKVCache(Fusion):
    """
    Fuse FusionSplitQKVUpdateKVCache
    """

    def __init__(self, model: OnnxModel, hidden_size: int, num_heads: int):
        super().__init__(
            model, "SplitQKVUpdateKVCache_IxRT", "Split"
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
        
        k_cache_output = outputs[1]
        v_cache_output = outputs[2]
        
        concat_k_input = k_cache_output + "_k_concat_input"
        concat_v_input = v_cache_output + "_v_concat_input"
        
        plugin_outputs = [outputs[0],concat_k_input,concat_v_input]
        
        new_node = helper.make_node(
            "SplitQKVUpdateKVCache_IxRT",
            inputs=inputs,
            outputs=plugin_outputs,
            name=node_name,
        )
        
        k_concat_node_name = node_name + "_k_concat"
        v_concat_node_name = node_name + "_v_concat"
        
        k_concat_node = helper.make_node(
            "Identity",
            inputs=[concat_k_input],
            outputs=[outputs[1]],
            name=k_concat_node_name,
        )
        

            
        v_concat_node = helper.make_node(
            "Identity",
            inputs=[concat_v_input],
            outputs=[outputs[2]],
            name=v_concat_node_name,
        )
        

        
        
        
        
        new_node.domain = "com.iluvatar"
        new_node.attribute.extend([helper.make_attribute("plugin_namespace", "")])
        new_node.attribute.extend([helper.make_attribute("plugin_version", "1")])
        new_node.attribute.extend([helper.make_attribute("num_head", self.num_heads)])
        new_node.attribute.extend(
            [helper.make_attribute("head_dim", self.hidden_size // self.num_heads)]
        )
        
        self.model.replace_input_of_all_nodes(outputs[1],concat_k_input)
        self.model.replace_input_of_all_nodes(outputs[2],concat_v_input)

        return new_node,k_concat_node,v_concat_node

    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        
        split_node = node
        split_data = self.model.get_initializer_input_edges(node.name,return_np_array = True)
        if split_data[0].shape != (3,):
            return 
        if split_data[0][0] != split_data[0][1] and  split_data[0][1] != split_data[0][2]:
            return

        q_input, k_input, v_input = node.output[0],node.output[1],node.output[2]  
              
        q_path_nodes= []
        k_path_nodes= []
        v_path_nodes= []
        
        reshape_nodes = self.model.get_children(node)
        
        for node in reshape_nodes:
            if node.op_type != "Reshape":
                return
        q_reshape_node,k_reshape_node,v_reshape_node =  reshape_nodes[0],reshape_nodes[1],reshape_nodes[2]   
                    
        q_path_nodes.append(q_reshape_node)
        k_path_nodes.append(k_reshape_node)    
        v_path_nodes.append(v_reshape_node) 
        
        q_transpose_nodes = self.model.get_children(q_reshape_node) 
        k_transpose_nodes = self.model.get_children(k_reshape_node) 
        v_transpose_nodes = self.model.get_children(v_reshape_node)
        
        if  len(q_transpose_nodes)!=1 and len(k_transpose_nodes) != 1 and len(v_transpose_nodes) != 1:
            return
        
        
        q_transpose_node = q_transpose_nodes[0]
        
        k_transpose_node = k_transpose_nodes[0]
        v_transpose_node = v_transpose_nodes[0]
        
        k_path_nodes.append(k_transpose_node)
        v_path_nodes.append(v_transpose_node)
        
        
        k_concat_nodes = self.model.get_children(k_transpose_node) 
        v_concat_nodes = self.model.get_children(v_transpose_node)
                
        if  len(k_transpose_nodes) != 1 or len(v_transpose_nodes) != 1:
            return
        
        k_concat_node = k_concat_nodes[0]
        v_concat_node = v_concat_nodes[0]
        
        if v_concat_node.attribute[0].i != 2 and k_concat_node.attribute[0].i != 2: #axis = 2
            return 
                
        k_path_nodes.append(k_concat_node)
        v_path_nodes.append(v_concat_node)
        
        k_cache_input = k_concat_node.input[0]
        if k_transpose_node.output[0] == k_concat_node.input[0]:
            k_cache_input = k_concat_node.input[1]
        k_cache_output =  k_concat_node.output[0]   
        
        
        
        v_cache_input = v_concat_node.input[0]
        if v_transpose_node.output[0] == v_concat_node.input[0]:
            v_cache_input = v_concat_node.input[1]
        v_cache_output =  v_concat_node.output[0]  
        
         
        plugin_inputs = [split_node.input[0],k_cache_input,v_cache_input] 
        plugin_outputs = [q_transpose_node.output[0], k_cache_output,v_cache_output]
        remove_nodes = [split_node, q_reshape_node,q_transpose_node]
        
        remove_nodes.extend(k_path_nodes)
        remove_nodes.extend(v_path_nodes)
        new_node,k_cache_concat_node, v_cache_concat_node= self.create_node(plugin_inputs, plugin_outputs)

        self.nodes_to_add.append(new_node)
        self.nodes_to_add.append(k_cache_concat_node)
        self.nodes_to_add.append(v_cache_concat_node)
        
        self.node_name_to_graph_name[new_node.name] = self.this_graph_name
        self.node_name_to_graph_name[k_cache_concat_node.name] = self.this_graph_name
        self.node_name_to_graph_name[v_cache_concat_node.name] = self.this_graph_name
        
        self.nodes_to_remove.extend(remove_nodes)
       
