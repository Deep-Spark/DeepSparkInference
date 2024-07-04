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

import onnx_graphsurgeon as gs
import onnx

onnx_op_set_2_ir_version = {
    11:6,
    12:7,
    13:7,
}

visited_add_tensor = {}
def replace_expand_values(graph, expand_node, clip_node, cast_node, sub_node, add_node):
    if add_node.inputs[0].name not in visited_add_tensor:
        print(add_node.inputs[0].name)
        print(add_node.inputs[0].values)
        add_node.inputs[0].values = add_node.inputs[0].values + 384
        add_node.inputs[0].values[add_node.inputs[0].values < 0] = 0
        add_node.inputs[0].values[add_node.inputs[0].values > 767] = 767
        print(add_node.inputs[0].values)
        visited_add_tensor[add_node.inputs[0].name] = True
    expand_node.inputs = [add_node.inputs[0]] + expand_node.inputs[1:]

def replace_clip_related_nodes(graph):
    node_name_to_index_map = {}
    expand_node_names = []
    output_name_to_node_name_map = {}
    for i, node in enumerate(graph.nodes):
        node_name_to_index_map[node.name] = i
        if node.op == "Expand":
            expand_node_names.append(node.name)
        for j in node.outputs:
            output_name_to_node_name_map[j.name] = node.name

    for name in expand_node_names:
        expand_node = graph.nodes[node_name_to_index_map[name]]
        expand_producer_name = output_name_to_node_name_map[expand_node.inputs[0].name]
        expand_producer = graph.nodes[node_name_to_index_map[expand_producer_name]]
        if expand_producer.op == "Clip":
            clip_node = expand_producer
            clip_producer_name = output_name_to_node_name_map[clip_node.inputs[-1].name]
            clip_producer = graph.nodes[node_name_to_index_map[clip_producer_name]]
            if  clip_producer.op == "Cast":
                cast_producer_name = output_name_to_node_name_map[clip_producer.inputs[0].name]
                cast_producer = graph.nodes[node_name_to_index_map[cast_producer_name]]
                if cast_producer.op == "Sub":
                    add_node_name = output_name_to_node_name_map[clip_node.inputs[0].name]
                    add_node = graph.nodes[node_name_to_index_map[add_node_name]]
                    replace_expand_values(graph, expand_node, clip_node, clip_producer, cast_producer, add_node)

def drop_cast_nodes(graph):
    node_name_to_index_map = {}
    cast_node_names = []
    output_name_to_node_name_map = {}
    for i, node in enumerate(graph.nodes):
        node_name_to_index_map[node.name] = i
        if node.op == "Cast":
            cast_node_names.append(node.name)
        for j in node.outputs:
            output_name_to_node_name_map[j.name] = node.name

    for name in cast_node_names:
        cast_node = graph.nodes[node_name_to_index_map[name]]
        cast_producer_name = output_name_to_node_name_map[cast_node.inputs[0].name]
        cast_producer = graph.nodes[node_name_to_index_map[cast_producer_name]]
        if cast_producer.op == "Cast":
            cast_node.inputs = cast_producer.inputs


input_path = r"/ixrt/deberta-torch-fp32-sim.onnx"
save_path = r"/ixrt/deberta-sim-drop-clip-drop-invaild-cast.onnx"
graph = gs.import_onnx(onnx.load(input_path))

replace_clip_related_nodes(graph)
drop_cast_nodes(graph)

graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), save_path)

model = onnx.load(save_path)
model.ir_version = onnx_op_set_2_ir_version[model.opset_import[0].version]
onnx.save(model, save_path)