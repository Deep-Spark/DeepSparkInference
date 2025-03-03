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

import argparse
import collections
import json
import os

import numpy as np
import onnx

def parse_onnx_model(onnx_model):
    graph = onnx_model.graph
    nodes = graph.node
    initializer = graph.initializer
    value_info = graph.value_info
    model = {}
    model["nodes"] = {}
    model["tensors"] = {}
    model["edges"] = {}
    all_edge = []
    for i, item in enumerate(nodes):
        node_name = item.name
        input_edge_list = list(item.input)
        output_edge_list = list(item.output)
        all_edge.extend(input_edge_list)
        all_edge.extend(output_edge_list)
        node_dict = {"inputs": input_edge_list, "outputs": output_edge_list}
        node_dict["op_type"] = item.op_type
        attribute_dict = {}
        for attr in item.attribute:
            if attr.type == onnx.AttributeProto().AttributeType.FLOAT:
                attribute_dict[attr.name] = attr.f
            if attr.type == onnx.AttributeProto().AttributeType.FLOATS:
                attribute_dict[attr.name] = [x for x in attr.floats]
            if attr.type == onnx.AttributeProto().AttributeType.INT:
                attribute_dict[attr.name] = attr.i
            if attr.type == onnx.AttributeProto().AttributeType.INTS:
                attribute_dict[attr.name] = [x for x in attr.ints]
            if attr.type == onnx.AttributeProto().AttributeType.STRING:
                attribute_dict[attr.name] = str(attr.s.decode("UTF-8"))
            if attr.type == onnx.AttributeProto().AttributeType.STRINGS:
                attribute_dict[attr.name] = [
                    str(x.decode("UTF-8")) for x in attr.strings
                ]
        node_dict["attrbiute"] = attribute_dict
        model["nodes"][node_name] = node_dict

    constant_edge = []
    for i, item in enumerate(initializer):
        tensor_name = item.name
        constant_edge.append(tensor_name)
        if item.data_type == 1:
            tensor_dict = {"data_type": "float32"}
        elif item.data_type == 3:
            tensor_dict = {"data_type": "int32"}
        elif item.data_type == 7:
            tensor_dict = {"data_type": "int64"}    
        tensor_dict["dims"] = list(item.dims)

        model["tensors"][tensor_name] = tensor_dict

    miss_edge = []
    for edge in all_edge:
        if edge not in constant_edge:
            miss_edge.append(edge)

    for info in value_info:
        info_name = info.name
        if info_name in miss_edge:
            edge_dict = {
                "dims": [int(x.dim_value) for x in info.type.tensor_type.shape.dim]
            }
            model["edges"][info_name] = edge_dict

    """
    Export weight
    """
    var_dict = collections.OrderedDict()
    for item in initializer:
        tensor_name = item.name
        tensor_shape = list(item.dims)
        if len(tensor_shape) == 0:
            continue

        if item.data_type == 1 and len(item.float_data):
            np_data = np.array(list(item.float_data), dtype=np.float32)
            np_data = np_data.reshape(tensor_shape)
            var_dict[tensor_name] = np_data
        elif item.data_type == 1 and len(item.raw_data):
            np_data = np.frombuffer(item.raw_data, dtype=np.float32)
            np_data = np_data.reshape(tensor_shape)
            var_dict[tensor_name] = np_data
        elif item.data_type == 3 and len(item.int32_data):
            np_data = np.array(list(item.int32_data), dtype=np.int32)
            np_data = np_data.reshape(tensor_shape)
            var_dict[tensor_name] = np_data
        elif item.data_type == 3 and len(item.raw_data):
            np_data = np.frombuffer(item.raw_data, dtype=np.int32)
            np_data.dtype = np.int32
            np_data = np_data.reshape(tensor_shape)
            var_dict[tensor_name] = np_data
        elif item.data_type == 7 and len(item.raw_data):   
            np_data = np.frombuffer(item.raw_data, dtype=np.int64)
            np_data = np_data.reshape(tensor_shape)
            var_dict[tensor_name] = np_data
        elif item.data_type == 7 and len(item.int64_data):
            temp = []
            for i in item.int64_data:
                temp.append(i)
            np_data = np.array(temp, dtype=np.int64)
            np_data = np_data.reshape(tensor_shape)
            var_dict[tensor_name] = np_data
        else:
            print(
                "tensor name: ",
                tensor_name,
                ", type: ",
                item.data_type,
                ", len: ",
                len(item.raw_data),
                len(item.float_data),
                len(item.int32_data),
                len(item.int64_data),
                ", will not save into weights file",
            )
    return model, var_dict