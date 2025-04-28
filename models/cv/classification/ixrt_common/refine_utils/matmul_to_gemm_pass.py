from refine_utils.common import *

#
#   Common pattern Matmul to Gemm
#
class FusedGemmPass(BasePass):

    def process(self, graph: Graph) -> Graph:
        self.transform = GraphTransform(graph)

        find_sequence_subgraph(
            graph, pattern=[OP.MATMUL], callback=self.to_gemm, strict=True
        )
        return graph

    def to_gemm(self, graph, pattern: PatternGraph):
        matmul_op = pattern.nodes[0]
        inputs = matmul_op.operator.inputs
        outputs = matmul_op.operator.outputs

        if len(inputs)!=2 and len(outputs)!=1:
            return

        for input in inputs:
            if self.transform.is_leaf_variable(input):
                return

        print(f"{self.transform.get_variable(inputs[0]).shape}   {self.transform.get_variable(inputs[1]).shape}")
        self.transform.delete_operator(matmul_op.operator)

        op = self.transform.make_operator(
            op_type = "Gemm",
            inputs = inputs,
            outputs = outputs,
            alpha = 1,
            beta = 1,
            transB = 1
        )

        self.transform.add_operator(op)