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
from typing import Union, Callable, List

from tensorrt.deploy.api import *
from tensorrt.deploy.backend.onnx.converter import default_converter
from tensorrt.deploy.backend.torch.executor.operators._operators import to_py_type
from tensorrt.deploy.ir.operator_attr import BaseOperatorAttr, EmptyAttr
from tensorrt.deploy.ir.operator_type import OperatorType as OP
from tensorrt.deploy.ir import operator_attr as attr, Operator, generate_operator_name
from tensorrt.deploy.fusion import BasePass, PatternGraph, build_sequence_graph, GraphMatcher, PassSequence
from tensorrt.deploy.ir import Graph
from tensorrt.deploy.quantizer.quant_operator.base import quant_single_input_operator
from tensorrt.deploy.backend.onnx.converter import convert_onnx_operator

def find_sequence_subgraph(graph,
                           pattern: Union[List[str], PatternGraph],
                           callback: Callable[[Graph, PatternGraph], None],
                           strict=True):
    if isinstance(pattern, List):
        pattern = build_sequence_graph(pattern)

    matcher = GraphMatcher(pattern, strict=strict)
    return matcher.findall(graph, callback)