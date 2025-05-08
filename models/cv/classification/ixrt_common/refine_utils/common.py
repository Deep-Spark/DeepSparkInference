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