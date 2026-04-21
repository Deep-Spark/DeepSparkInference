import os
import cv2
import argparse
import numpy as np

import torch
import tensorrt

from tensorrt import Dims


def main(config):
    
    input_shape = [args.batch_size,3, 736,1280]
    IXRT_LOGGER = tensorrt.Logger(tensorrt.Logger.WARNING)
    builder = tensorrt.Builder(IXRT_LOGGER)
    EXPLICIT_BATCH = 1 << (int)(tensorrt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)
    build_config = builder.create_builder_config()
    parser = tensorrt.OnnxParser(network, IXRT_LOGGER)
    parser.parse_from_file(config.model)

    precision = tensorrt.BuilderFlag.INT8 if config.precision == "int8" else tensorrt.BuilderFlag.FP16
    build_config.set_flag(precision)
    if config.precision == "int8":
        build_config.set_flag(tensorrt.BuilderFlag.FP16)
    
    input_tensor = network.get_input(0)
    input_tensor.shape = Dims(input_shape)

    plan = builder.build_serialized_network(network, build_config)
    engine_file_path = config.engine
    with open(engine_file_path, "wb") as f:
        f.write(plan)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,default="wide_deep.onnx")
    parser.add_argument("--precision", type=str, choices=["float16", "int8", "float32"], default="float16",
            help="The precision of datatype")
    parser.add_argument("--engine", type=str, default="wide_deep.engine")
    parser.add_argument("--batch_size", type=int, default=1)
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)