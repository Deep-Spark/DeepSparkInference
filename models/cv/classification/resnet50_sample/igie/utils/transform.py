import tvm  # type: ignore
from tvm import relay
from tvm.relay import Expr
from tvm.relay.frontend.common import infer_shape
from tvm.relay.dataflow_pattern import DFPatternCallback
from tvm.relay.dataflow_pattern import wildcard
from tvm.relay.dataflow_pattern import is_op
from tvm.relay.expr_functor import ExprMutator

class Simplify_Rewriter(DFPatternCallback):
    '''
    %5 = nn.pad(%4, 0 /* ty=int32 */, pad_width=[[0, 0], [0, 0], [0, 0], [0, 1]]) /* ty=Tensor[(32, 224, 224, 4), int8] */;
    %6 = nn.qdconv(%5, meta[relay.Constant][0] /* ty=Tensor[(64, 7, 7, 4), int8] */, meta[relay.Constant][1] /* ty=Tensor[(1, 1, 1, 64), float32] */, strides=[2, 2], padding=[3, 3, 3, 3], alpha=0.00291618f, channels=64, kernel_size=[7, 7], act_func="relu", connect_mode="", out_dtype="int8") /* ty=Tensor[(32, 112, 112, 64), int8] */;
    '''
    def __init__(self, require_type=False, rewrite_once=False):
        super().__init__(require_type, rewrite_once)
        self.data = wildcard()
        self.weight = wildcard()
        self.bias = wildcard()
        self.pad = is_op("nn.pad")(self.data, wildcard())
        self.qdconv = is_op("nn.qdconv")(self.pad, self.weight, self.bias)
        self.pattern = self.qdconv

    def callback(self, pre: Expr, post: Expr, node_map: tvm.ir.container.Map) -> Expr:
        data = node_map[self.data][0]
        weight = node_map[self.weight][0]
        bias = node_map[self.bias][0]
        conv_attrs = node_map[self.qdconv][0].attrs
        shape = infer_shape(data)
        if shape[0] != 8:
            return post
        else:
            pad = relay.nn.pad(data, pad_width=[[0, 8], [0, 0], [0, 0], [0, 1]])
        qdconv = relay.nn.qdconv(pad, weight, bias, alpha=conv_attrs.alpha, strides=conv_attrs.strides,
                               padding=conv_attrs.padding, dilation=conv_attrs.dilation, groups=conv_attrs.groups, channels=conv_attrs.channels,
                               kernel_size=conv_attrs.kernel_size, act_func=conv_attrs.act_func, data_layout=conv_attrs.data_layout, kernel_layout=conv_attrs.kernel_layout,
                                 out_layout=conv_attrs.out_layout, out_dtype=conv_attrs.out_dtype)
        
        return qdconv