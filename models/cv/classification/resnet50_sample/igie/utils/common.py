import argparse
import os
import numpy as np
import tvm
from tvm import relay
from tvm.relay import Expr
from tvm.relay.transform.iluvatar.optimize_graph import dump_mod_to_file
from tvm.relay.transform.iluvatar.legalize import SimplifyModuleFirstOp
from tvm.relay.dataflow_pattern import wildcard, is_constant, is_op, DFPatternCallback, rewrite
from tvm.relay.expr_functor import ExprMutator

def str2bool(str):
    return True if str.lower() == 'true' else False

def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='Classification Inferecne', add_help=add_help)

    parser.add_argument('--model-name', type=str, default='model', help='Model name.')
    parser.add_argument('--model-path', type=str, required=False, help='Model checkpoint path.')
    parser.add_argument('--model-format', type=str, choices=['onnx', 'pytorch', 'tensorflow'], required=True, help='Checkpoint format.')
    parser.add_argument('--model-layout', type=str, choices=['NCHW', 'NHWC'], default="NCHW", help='Model inference data layout.')
    
    parser.add_argument('--data-path', type=str, required=False, help='Inference dataset image path.')
    parser.add_argument('--label-path', type=str, required=False, help='Inference dataset label path.')
    
    parser.add_argument('--input-name', type=str, required=True, help='Model input name.')
    parser.add_argument('--input-shape', type=int, nargs='*', default=[], required=True, help='use --input-shape 16 3 224 224 to set model input shape.')
    parser.add_argument('--precision', type=str, choices=['int8', 'fp16', 'fp32'], required=True, help='Model inference precision.')
    parser.add_argument('--num-classes', type=int, default=1000, help='model num classes.')
    
    parser.add_argument('--target', type=str, choices=['llvm', 'iluvatar', 'iluvatar_libs'], default="iluvatar_libs", help='IGIE target')
    parser.add_argument('--warmup', type=int, default=5, help='Number of queries to warmup before test.')
    parser.add_argument('--igie-quantize', action='store_true', help='use igie-quantization inference')
    parser.add_argument('--onnx-quantize', action='store_true', help='use igie-quantization inference')
    parser.add_argument('--convert-layout', type=str, choices=['NCHW', 'NHWC'], default="NHWC", help='Model inference data format.')
    parser.add_argument('--export-engine', type=str2bool, default=True,  help='Export Model Engine.')
    parser.add_argument('--use-ixinfer', type=str2bool, default=True, help='use ixinfer fused op for model inference.')
    parser.add_argument('--with-nms', action='store_true', help='Only for YOLO model inference with NMS op.')
    parser.add_argument('--workers', default=0, type=int, metavar='N', help='number of data loading workers (default: 0)')

    parser.add_argument('--verbose', type=str2bool, default=False,  help='dump igie mod to file.')
    parser.add_argument('--test-count', type=int, default=1000, help='Sample count to test.')
    parser.add_argument('--acc1-target', type=float, default=0, help='Model inference Top1 Accuracy target.')
    parser.add_argument('--acc5-target', type=float, default=0, help='Model inference Top5 Accuracy target.')
    parser.add_argument('--fps-target', type=float, default=0, help='Model inference FPS target.')
    parser.add_argument('--latency-target', type=float, default=0, help='Model inference QPS target.')
    parser.add_argument('--print-freq', type=int, default=1, help='Print log frequency')
    return parser

### create runtime target.
def get_target(args):
    target_name = args.target
    precision = args.precision
    use_ixinfer = args.use_ixinfer
    
    target = None
    if target_name == "iluvatar":
        target = tvm.target.iluvatar(model="MR")
    
    elif target_name == "iluvatar_libs":
        if use_ixinfer == True:
            target = tvm.target.iluvatar(model="MR", options="-libs=cudnn,cublas,ixinfer")
        else:
            target = tvm.target.iluvatar(model="MR", options="-libs=cudnn,cublas")
            
    elif target_name == "llvm":
        target = tvm.target.Target(target_name)
    else:
        raise Exception(f"Unsupport Target name: {target_name}!")
    
    device = tvm.device(target.kind.name, 0)
    
    return target, device


def get_input_shape(set_input_shape):
    ## 指定input_shape
    input_shape = []
    if len(set_input_shape) != 0:
        for i in set_input_shape:
            input_shape.append(i)
        return input_shape
    else:
        raise ValueError("Input Shape is None, Please use 'e.g --input-shape 16 3 224 224 ' to set model input shape.")


def get_file_path(args, path_dir):
    if not os.path.isdir(path_dir):
        os.makedirs(path_dir)
        
    if "engine" in path_dir:
        export_engine_path = path_dir + "/export_engine_%s_%s_%s_%s_batch%d_%s.so" % (args.model_format, args.model_name, args.convert_layout, args.precision, args.batch_size, "ixinfer" if args.use_ixinfer else "igie")
        
        # for yolo model
        if args.with_nms:
            export_engine_path = export_engine_path.split(".so")[0] + "_with_nms.so"
        return export_engine_path
    
    elif "log" in path_dir:
        tune_log_file = path_dir + "/tune_%s_%s_%s_%s_batch%d.log" % (args.model_format, args.model_name, args.convert_layout, args.precision, args.batch_size)
        return tune_log_file


def check_status(args, benchmark_result):
    # return a status to system, shell can use "echo $?" to get the status,
    # 0 means task success and 1 means task failed.
    if (args.acc1_target == 0 or benchmark_result["acc@1"] >= args.acc1_target) \
            and (args.acc5_target == 0 or benchmark_result["acc@5"] >= args.acc5_target) \
            and (args.fps_target == 0 or benchmark_result["fps"] >= args.fps_target) \
            and (args.latency_target == 0 or benchmark_result["latency"] >= args.latency_target):
        print("Test Finish!")
        return 0
    else:
        if (benchmark_result["acc@1"] < args.acc1_target):
            print("Expected accuracy Acc@1: %f, actual inference accuracy Acc@1: %f " % (args.acc1_target, benchmark_result["acc@1"]))
        
        if (benchmark_result["acc@5"] < args.acc5_target):
            print("Expected accuracy Acc@5: %f, actual inference accuracy Acc@5: %f " % (args.acc5_target, benchmark_result["acc@5"]))

        if (benchmark_result["fps"] < args.fps_target):
            print("Expected FPS: %f, actual inference FPS: %f " % (args.fps_target, benchmark_result["fps"]))
            
        if (benchmark_result["latency"] > args.latency_target and args.latency_target != 0):
            print("Expected Latency: %f, actual inference Latency: %f " % (args.latency_target, benchmark_result["latency"]))
        
        print("\n====Test failed!====\n")
        return 1
    
def list_ops(expr):
    class OpLister(tvm.relay.ExprVisitor):
        def visit_op(self, expr):
            if expr not in self.node_set:
                self.node_list.append(expr)
            return super().visit_op(expr)

        def list_nodes(self, expr):
            self.node_set = {}
            self.node_list = []
            self.visit(expr)
            return self.node_list

    return OpLister().list_nodes(expr)


def first_layer_transform(mod, input_name="data", input_shape=[], dtype="int8", verbose=False):
    if input_shape[0] == 8:
        return mod
    mod = SimplifyModuleFirstOp(input_name=input_name, input_shape=input_shape, dtype=dtype)(mod)
    dump_mod_to_file(mod, "SimplifyModuleFirstOp.log", mod_folder="modules", verbose=verbose)
    return mod

def first_layer_process(image, layout="NCHW", input_scales=48.4848):
    ## NCHW -> NHWC
    image = np.array(image)
    if image.shape[0] == 8:
        return image
    if layout == "NCHW":
        image = np.transpose(image, [0, 2, 3, 1])
    image = image * input_scales
    image = np.pad(image, pad_width=((0, 0), (0, 0), (0, 0), (0, 1)), mode="constant")
    image = image.astype("int8")
    return image

class MainFunctionParamsRewriter(ExprMutator):
    def __init__(self, target_input_dict, preprocess_rewriter):
        self.target_input = target_input_dict
        self.preprocess_rewriter = preprocess_rewriter
        self.target_input_name_list = list(self.target_input.keys())

        super().__init__()

    def visit_function(self, fn):
        params = [self.visit(i) for i in fn.params]
        body  = self.visit(fn.body)

        original_input_name_list = [param.name_hint for param in params]
        assert len(set(self.target_input_name_list) - set(original_input_name_list)) == 0, f"invalid target_input_name: {set(self.target_input_name_list) - set(original_input_name_list)}"

        new_params = []
        bind = {}
        for param in params:
            old_param = param
            name = param.name_hint

            new_param = old_param
            if name in self.target_input:
                shape = self.target_input[name][0]
                dtype = self.target_input[name][1]
                new_param = relay.var(name_hint=name, shape=shape, dtype=dtype)

            new_params.append(new_param)
            bind[old_param] = new_param

        new_body = relay.expr.bind(body, bind)

        new_function = relay.Function(params=new_params,
                                      body=new_body,
                                      ret_type=None,
                                      type_params=fn.type_params,
                                      attrs=fn.attrs)
        return new_function

    def __call__(self, mod):
        mod["main"] = rewrite(self.preprocess_rewriter, mod["main"])
        mod["main"] = self.visit(mod["main"])
        return mod

class ResNetFloat16PreprocessRewriter(DFPatternCallback):
    def __init__(self, require_type=False, rewrite_once=False):
        super().__init__(require_type, rewrite_once)
        """
            %0 = cast(%data, dtype="float16") /* ty=Tensor[(32, 3, 224, 224), float16] */;
            %1 = layout_transform(%0, src_layout="NCHW", dst_layout="NHWC") /* ty=Tensor[(32, 224, 224, 3), float16] */;
            %2 = nn.pad(%1, 0 /* ty=int32 */, pad_width=[[0, 0], [0, 0], [0, 0], [0, 1]]) /* ty=Tensor[(32, 224, 224, 4), float16] */;
        """
        self.var_images = wildcard()
        self.const_0 = is_constant()  # 0 /* ty=int32 */
        self.cast_0 = is_op('cast')(self.var_images)  # /* ty=Tensor[(32, 3, 224, 224), float16] */
        self.layout_transform_1 = is_op('layout_transform')(self.cast_0)  # /* ty=Tensor[(32, 224, 224, 3), float16] */
        self.nn_pad_2 = is_op('nn.pad')(self.layout_transform_1, self.const_0)  # /* ty=Tensor[(32, 224, 224, 4), float16] */
        self.pattern = self.nn_pad_2


    def callback(self, pre: Expr, post: Expr, node_map: tvm.ir.container.Map) -> Expr:
        return node_map[self.var_images][0]