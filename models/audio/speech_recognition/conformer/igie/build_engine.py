import os
import json
import onnx
import logging
import argparse

import tensorrt
from tensorrt import Dims
import tvm
from tvm import relay
from tvm.relay.import_model import import_model_to_igie

from load_ixrt_plugin import load_ixrt_plugin
load_ixrt_plugin()

def parse_args():
    parser = argparse.ArgumentParser(description="Build tensorrt engine of deepspeech2")
    parser.add_argument("--onnx_model", type=str, required=True, help="The onnx path")
    parser.add_argument("--bsz", type=int, default=1, help="batch size")
    parser.add_argument("--input_size", type=tuple, default=(-1, 161), help="inference size")
    parser.add_argument("--engine_path", type=str, required=True, help="engine path to save")
    parser.add_argument( "--device", type=int, default=0, help="cuda device, i.e. 0 or 0,1,2,3,4")

    args = parser.parse_args()
    return args


def build_engine_trtapi_dynamicshape(args):
    onnx_model = args.onnx_model
    assert os.path.isfile(onnx_model), f"The onnx model{onnx_model} must be existed!"
    IXRT_LOGGER = tensorrt.Logger(tensorrt.Logger.WARNING)
    builder = tensorrt.Builder(IXRT_LOGGER)
    EXPLICIT_BATCH = 1 << (int)(tensorrt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)
    build_config = builder.create_builder_config()
    profile = builder.create_optimization_profile()
    
    profile.set_shape(
        "input", Dims([1,1,80]),Dims([16,800,80]),Dims([128,1500,80])
    )
    profile.set_shape(
        "seq_lengths", Dims([1]), Dims([16]), Dims([128])
    )

    build_config.add_optimization_profile(profile)

    parser = tensorrt.OnnxParser(network, IXRT_LOGGER)

    parser.parse_from_file(onnx_model)
    build_config.set_flag(tensorrt.BuilderFlag.FP16)

    # set dynamic
    input_tensor = network.get_input(0)
    input_tensor.shape = Dims([-1, -1, 80])
    
    seq_lengths_tensor = network.get_input(1)
    seq_lengths_tensor.shape = Dims([-1])

    plan = builder.build_serialized_network(network, build_config)
    with open(args.engine_path, "wb") as f:
        f.write(plan)

    print("Build dynamic shape engine done!")



def build_engine_igieapi_dynamicshape(args):
    onnx_model = args.onnx_model
    assert os.path.isfile(onnx_model), f"The onnx model{onnx_model} must be existed!"
    
    target = tvm.target.iluvatar(model="MR", options="-libs=cudnn,cublas,ixinfer")
    device = tvm.device(target.kind.name, 0)
    inputs_info = {'input': ([128, 1500, 80], 'float16'), 'seq_lengths': ([128], 'int32')}
    precision = "fp16"

    mod, params = import_model_to_igie(onnx_model, inputs_info, outputs_info=None, precision=precision, backend="tensorrt")
    lib = relay.build(mod, target=target, params=params, precision=precision, device=device)
    lib.export_library(args.engine_path)

    print("Build dynamic shape engine done!")
    
if __name__ == "__main__":
    args = parse_args()
    build_engine_trtapi_dynamicshape(args)

