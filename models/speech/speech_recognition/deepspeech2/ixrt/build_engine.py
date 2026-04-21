import os
import json
import onnx
import logging
import argparse
import ctypes
import tensorrt
from tensorrt import Dims
from load_ixrt_plugin import load_ixrt_plugin

load_ixrt_plugin()

def parse_args():
    parser = argparse.ArgumentParser(description="Build tensorrt engine of deepspeech2")
    parser.add_argument("--model_name", type=str, required=True, help="model name deepspeech2")
    parser.add_argument("--onnx_path", type=str, required=True, help="The onnx path")
    parser.add_argument("--bsz", type=int, default=1, help="batch size")
    parser.add_argument("--input_size", type=tuple, default=(-1, 161), help="inference size")
    parser.add_argument("--engine_path", type=str, required=True, help="engine path to save")
    parser.add_argument( "--device", type=int, default=0, help="cuda device, i.e. 0 or 0,1,2,3,4")

    args = parser.parse_args()
    return args


def build_engine_trtapi_dynamicshape(args):
    onnx_model = args.onnx_path
    assert os.path.isfile(onnx_model), f"The onnx model{onnx_model} must be existed!"
    IXRT_LOGGER = tensorrt.Logger(tensorrt.Logger.WARNING)
    builder = tensorrt.Builder(IXRT_LOGGER)
    EXPLICIT_BATCH = 1 << (int)(tensorrt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)
    build_config = builder.create_builder_config()

    profile = builder.create_optimization_profile()

    profile.set_shape(
        "input", Dims([1, 100, 161]), Dims([1, 1193, 161]), Dims([1, 3494, 161])
    )

    build_config.add_optimization_profile(profile)

    parser = tensorrt.OnnxParser(network, IXRT_LOGGER)

    parser.parse_from_file(onnx_model)
    build_config.set_flag(tensorrt.BuilderFlag.FP16)

    # set dynamic
    input_tensor = network.get_input(0)
    input_tensor.shape = Dims([1, -1, 161])

    plan = builder.build_serialized_network(network, build_config)
    with open(args.engine_path, "wb") as f:
        f.write(plan)

    print("Build dynamic shape engine done!")


def build_engine_trtapi_staticshape(args):
    onnx_model = args.onnx_path
    assert os.path.isfile(onnx_model), f"The onnx model{onnx_model} must be existed!"
    IXRT_LOGGER = tensorrt.Logger(tensorrt.Logger.WARNING)
    builder = tensorrt.Builder(IXRT_LOGGER)
    EXPLICIT_BATCH = 1 << (int)(tensorrt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)
    build_config = builder.create_builder_config()
    parser = tensorrt.OnnxParser(network, IXRT_LOGGER)

    parser.parse_from_file(onnx_model)
    build_config.set_flag(tensorrt.BuilderFlag.FP16)

    plan = builder.build_serialized_network(network, build_config)
    with open(args.engine_path, "wb") as f:
        f.write(plan)

    print("Build static shape engine done!")


if __name__ == "__main__":
    args = parse_args()
    build_engine_trtapi_dynamicshape(args)
    # build_engine_trtapi_staticshape(args)
