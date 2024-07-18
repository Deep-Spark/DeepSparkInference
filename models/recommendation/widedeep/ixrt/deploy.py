import onnx
import argparse
import copy

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
from tensorrt.deploy.api import GraphTransform, create_source, create_target

class FuseGemmPass(BasePass):
    def process(self, graph: Graph) -> Graph:
        self.transform = GraphTransform(graph)

        self.transform.find_sequence_subgraph(
            pattern=[OP.MATMUL, OP.ADD], callback=self.fuse_gemm, strict=True
        )
        return graph

    def fuse_gemm(self, graph, pattern: PatternGraph):
        matmul = pattern.nodes[0]
        add = pattern.nodes[1]

        if len(add.operator.inputs) != 2:
            return

        b_var = graph.get_variable(matmul.operator.inputs[1])
        if not graph.is_leaf_variable(b_var) or b_var.value is None:
            return

        if b_var.value.ndim != 2:
            return

        bias_var = None
        for input in add.operator.inputs:
            if input not in matmul.operator.outputs:
                bias_var = input

        matmul.operator.inputs.append(bias_var)
        self.transform.delete_operator_and_link(
            add.operator, link_input=matmul.operator.outputs[0]
        )

        matmul.operator.op_type = OP.GEMM
        matmul.operator.attributes = attr.GemmAttr(transB=1)

def replace_input(graph):
    transformer = GraphTransform(graph)
    from_op = graph.get_operator("Shape__8")
    to_op = graph.get_operator('import/head/predictions/zeros_like')
    var = graph.get_variable("import/head/predictions/zeros_like:0")
    transformer.delete_operators_between_op_op(from_op=from_op, to_op=to_op)
    transformer.add_input("import/head/predictions/zeros_like:0")
    return graph


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="")
    parser.add_argument("--output_path", default="")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    graph = create_source(args.model_path)()
    graph = FuseGemmPass().process(graph)
    graph = replace_input(graph)
    create_target(saved_path=args.output_path).export(graph)