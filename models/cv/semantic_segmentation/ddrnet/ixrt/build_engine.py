import os
import json
import onnx
import logging
import argparse
import ctypes
from os.path import join, dirname, exists

import tensorrt

def load_ixrt_plugin(logger=tensorrt.Logger(tensorrt.Logger.INFO), namespace="", dynamic_path=""):
    if not dynamic_path:
        dynamic_path = join(dirname(tensorrt.__file__), "lib", "libixrt_plugin.so")
    if not exists(dynamic_path):
        raise FileNotFoundError(
            f"The ixrt_plugin lib {dynamic_path} is not existed, please provided effective plugin path!")
    ctypes.CDLL(dynamic_path)
    tensorrt.init_libnvinfer_plugins(logger, namespace)
    print(f"Loaded plugin from {dynamic_path}")

load_ixrt_plugin()


def build_engine_trtapi(config):
    onnx_model = config.model
    assert os.path.isfile(onnx_model), f"The onnx model{onnx_model} must be existed!"
    IXRT_LOGGER = tensorrt.Logger(tensorrt.Logger.WARNING)
    builder = tensorrt.Builder(IXRT_LOGGER)
    EXPLICIT_BATCH = 1 << (int)(tensorrt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)
    build_config = builder.create_builder_config()
    parser = tensorrt.OnnxParser(network, IXRT_LOGGER)

    parser.parse_from_file(onnx_model)
    if config.precision == "int8":
        build_config.set_flag(tensorrt.BuilderFlag.INT8)
        build_config.set_flag(tensorrt.BuilderFlag.FP16)
    else: 
        build_config.set_flag(tensorrt.BuilderFlag.FP16)

    plan = builder.build_serialized_network(network, build_config)
    with open(config.engine, "wb") as f:
        f.write(plan)

    print("Build engine done!")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,  default="ddrnet23.onnx")
    parser.add_argument("--bsz", type=int, default=4, help="batch size")
    parser.add_argument("--precision", type=str, choices=["float16", "int8"], default="int8", help="The precision of datatype")
    parser.add_argument("--imgsz_h", type=int, default=1024, help="inference size h")
    parser.add_argument("--imgsz_w", type=int, default=2048, help="inference size w")
    # engine args
    parser.add_argument("--engine", type=str, default=None)
    # device
    parser.add_argument(
        "--device", type=int, default=0, help="cuda device, i.e. 0 or 0,1,2,3,4"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    config = parse_args()
    build_engine_trtapi(config)
