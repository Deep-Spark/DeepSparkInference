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

import onnx
from onnx import numpy_helper
import numpy as np
import os
import argparse

def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='alter onnx model', add_help=add_help)

    parser.add_argument('--batch_size', type=int, default=24, help='Model batch size.')
    parser.add_argument('--path', type=str, required=True, help='ONNX model path.')
    return parser


args = get_args_parser().parse_args()

encoder_onnx_path=args.path
batch_size = args.batch_size
onnx_model = onnx.load(encoder_onnx_path)


graph = onnx_model.graph
node  = graph.node

matmul_input_node = []
for i in range(len(node)):
    if node[i].op_type == 'MatMul':
        for name in node[i].input:
            matmul_input_node.append(name)

## alter node
for initializer in graph.initializer:
    if initializer.name in matmul_input_node:
        if initializer.dims[0] == 1:
            W = numpy_helper.to_array(initializer)
            W_new  = []
            for i in range(batch_size):
                W_new.append(W[0])
            W_new = np.array(W_new)
            tensor = numpy_helper.from_array(W_new, initializer.name)
            initializer.CopyFrom(tensor)
            initializer.dims[0] = batch_size

## print node
for initializer in graph.initializer:
    if initializer.name in matmul_input_node:
        if initializer.dims[0] == 24:
            W = numpy_helper.to_array(initializer)
            weights_map = {}
            weights_map[initializer.name] = W

onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
onnx.checker.check_model(onnx_model)

file_name, file_ext = os.path.splitext(encoder_onnx_path)
print("Save New Model to ", file_name + "_matmul.onnx")
onnx.save(onnx_model, file_name + "_matmul.onnx")
