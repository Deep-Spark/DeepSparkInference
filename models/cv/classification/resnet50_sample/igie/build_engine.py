import os
from pprint import pprint

import tvm
from tvm import relay
from tvm.relay.transform.iluvatar.optimize_graph import dump_mod_to_file

import logging
logging.basicConfig(level=logging.INFO, format = '[%(asctime)s %(filename)s line:%(lineno)d] %(levelname)s: %(message)s')
logging.getLogger('autotvm').setLevel(logging.INFO)
logging.getLogger('te_compiler').setLevel(logging.ERROR)

from utils.common import get_args_parser, get_input_shape, get_target, get_file_path, first_layer_transform
from utils.transform import Simplify_Rewriter
from tvm.relay.dataflow_pattern import rewrite
from tvm.relay.import_model import import_model_to_igie
from tvm.relay.transform.iluvatar.optimize_graph import convert_graph_layout_simplify

from utils.quantize import igie_quantize_model
from utils.common import MainFunctionParamsRewriter, ResNetFloat16PreprocessRewriter

def main():
    args = get_args_parser().parse_args()
    try:
        from dltest import show_infer_arguments
        show_infer_arguments(args)
    except:
        pass

    ### 设置模型初始信息
    input_name = args.input_name
    input_shape = get_input_shape(args.input_shape)
    args.batch_size = input_shape[0]
    
    print("Run Model args: ")
    pprint(vars(args), indent=2)
    
    ### 创建Target设备.
    target, device = get_target(args)
    
    ### 打印log信息
    verbose = args.verbose

    precision = args.precision

    custom_pass = None
    
    ### 设置engine文件path
    engine_dir = os.getcwd() + "/engine"
    export_engine_path = get_file_path(args, engine_dir)
    
    ### 如果存在engine，直接load engine进行推理.
    if os.path.isfile(export_engine_path):
        logging.info("\n\nUsing exported engine: %s to inference...\n", export_engine_path)      
    ### 导入模型 & 量化 & 编译
    else:
        ### 1. 导入预训练的模型到IGIE.
        input_dict = {input_name: input_shape}
        mod, params = import_model_to_igie(args.model_path, input_dict, precision=precision, backend="igie")
        mod = convert_graph_layout_simplify(mod, convert_layout=args.convert_layout)
        
        ### 2. 模型量化
        if args.igie_quantize:
            mod, params = igie_quantize_model(mod, params, args, scale_file_path="")
            dump_mod_to_file(mod, "quantize_mod.log", mod_folder="modules", verbose=verbose)
            
        ### 用户自定义一些图优化函数，例如：对模型输入进行处理。如果没有自定义的Pass，可以返回None.
        if precision == "int8":
            def custom_pass(mod):
                mod = first_layer_transform(mod, input_name="data", input_shape=[args.batch_size, 224, 224, 4], dtype="int8")
                mod["main"] = rewrite(Simplify_Rewriter(), mod["main"])
                mod = relay.transform.InferType()(mod)
                return mod
            
        if precision == "fp16":
            def custom_pass(mod):
                r50_preprocessor = MainFunctionParamsRewriter(
                    target_input_dict={input_name: [(args.batch_size, 224, 224, 4), "float16"]},
                    preprocess_rewriter=ResNetFloat16PreprocessRewriter()
                )
                mod = r50_preprocessor(mod)
                return mod
        ### 3. 编译模型
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target, params=params, custom_pass=custom_pass, verbose=verbose, precision=precision)
            dump_mod_to_file(lib.function_metadata["__tvm_main__"], "lib_final_mod.log", mod_folder="modules", verbose=verbose)
            
            ### 导出编译好的模型engine, 方便下次加载
            if args.export_engine:
                lib.export_library(export_engine_path)
                logging.info(f"export engine path is {export_engine_path}")
        
if __name__ == '__main__':
    main()
