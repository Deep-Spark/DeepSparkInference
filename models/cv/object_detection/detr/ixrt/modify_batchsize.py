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
from copy import deepcopy
import numpy as np
import onnx
from onnx import numpy_helper

from extract_graph_weight import parse_onnx_model
            
            
def modify_shape_dim(dim, bsz):
    batch_size = bsz
    # update dim to be a symbolic value
    if isinstance(batch_size, str):
        # set dynamic batch size
        dim.dim_param = batch_size
    elif (isinstance(batch_size, str) and batch_size.isdigit()) or isinstance(batch_size, int):
        # set given batch size
        dim.dim_value = int(batch_size)
    else:
        # set batch size of 1
        dim.dim_value = 1

def change_input_dim(onnx_model, bsz):
    inputs = onnx_model.graph.input
    for input in inputs:
        dim1 = input.type.tensor_type.shape.dim[0]
        old_bsz = dim1.dim_value
        modify_shape_dim(dim1, bsz)
        return old_bsz

# input[1] shape is initializer
def change_reshape_initializer(model, var_dict, old_bsz, bsz):
    print("change_reshape_initializer")
    modified_list = list()
    for name, node_dict in model["nodes"].items():
        if node_dict["op_type"] != "Reshape":
            continue
        shape_name = node_dict["inputs"][1]
        new_datas = deepcopy(var_dict[shape_name])
        done = False
        if (len(new_datas) == 2):
            if new_datas[0] == 625:
                new_datas[0] = 625 * (bsz / old_bsz)
            if new_datas[0] / old_bsz == 100:
                new_datas[0] = 100 * bsz
        elif (len(new_datas) == 3):
            for i in range(len(new_datas)):  
                if new_datas[i] == old_bsz:
                    new_datas[i] = bsz
                    done = True
            if done == False:
                for i in range(len(new_datas)):  
                    if new_datas[i] / old_bsz == 8:
                        new_datas[i] = (bsz / old_bsz) * 8
                        done = True
            
        var_dict[shape_name] = new_datas
        modified_list.append(shape_name)
    return modified_list

def change_matmul_initializer(model, var_dict, bsz):
    print("change_matmul_initializer")
    modified_list = list()
    for name, node_dict in model["nodes"].items():
        if node_dict["op_type"] != "MatMul":
            continue
        for edge_name in node_dict["inputs"]:
            if edge_name not in var_dict:
                continue
            if len(var_dict[edge_name].shape) != 3:
                continue
            data = deepcopy(var_dict[edge_name])

            datas = list()
            for _ in range(bsz):
                datas.append(data)
            new_datas = np.concatenate(datas, axis=0)
            var_dict[edge_name] = new_datas
            modified_list.append(edge_name)
    return modified_list

def change_add_initializer(model, var_dict, bsz):
    print("change_add_initializer")
    modified_list = list()
    for name, node_dict in model["nodes"].items():
        if node_dict["op_type"] != "Add":
            continue
        for edge_name in node_dict["inputs"]:
            if edge_name not in var_dict:
                continue
            if len(var_dict[edge_name].shape) != 3:
                continue
            data = deepcopy(var_dict[edge_name])[:, 0:1, ...]

            datas = list()
            for _ in range(bsz):
                datas.append(data)
            new_datas = np.concatenate(datas, axis=1)
            var_dict[edge_name] = new_datas
            modified_list.append(edge_name)
    return modified_list

# A certain mode, input for Concat operator maybe constant.
def change_concat_initializer(model, var_dict, bsz):
    print("change_concat_initializer")
    modified_list = list()
    for name, node_dict in model["nodes"].items():
        if node_dict["op_type"] != "Concat":
            continue
        for edge_name in node_dict["inputs"]:
            if edge_name not in var_dict:
                continue
            data = deepcopy(var_dict[edge_name])[0:1, ...]

            datas = list()
            for _ in range(bsz):
                datas.append(data)
            new_datas = np.concatenate(datas, axis=0)
            var_dict[edge_name] = new_datas
            modified_list.append(edge_name)
    return modified_list

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--origin_model", type=str)
    parser.add_argument("--output_model", type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    onnx_model = onnx.load(args.origin_model)
    bsz = args.batch_size
    old_bsz = change_input_dim(onnx_model, bsz)
    if old_bsz == bsz:
        print("Change batch size skipped")
        onnx.save(onnx_model, args.output_model)
        exit()

    model, weights = parse_onnx_model(onnx_model)

    modified_list = list()
    reshape_modified = change_reshape_initializer(model, weights, old_bsz, bsz)
    concat_modified = change_concat_initializer(model, weights, bsz)
    matmul_modified = change_matmul_initializer(model, weights, bsz)
    add_modified = change_add_initializer(model, weights, bsz)
    modified_list.extend(reshape_modified)
    modified_list.extend(concat_modified)
    modified_list.extend(matmul_modified)
    modified_list.extend(add_modified)

    # Remove the old initializer, and append new.
    initializer = onnx_model.graph.initializer
    for name in modified_list:
        for item in initializer:
            if name == item.name:
                initializer.remove(item)

        data = weights[name]
        new_params = numpy_helper.from_array(data, name=name)
        initializer.append(new_params)

    onnx.save(onnx_model, args.output_model)