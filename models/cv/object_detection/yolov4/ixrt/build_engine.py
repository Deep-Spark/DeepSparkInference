import os
import cv2
import argparse
import numpy as np

import torch
import tensorrt
from tensorrt import Dims

from load_ixrt_plugin import load_ixrt_plugin
load_ixrt_plugin()

def main(config):
    IXRT_LOGGER = tensorrt.Logger(tensorrt.Logger.WARNING)
    builder = tensorrt.Builder(IXRT_LOGGER)
    EXPLICIT_BATCH = 1 << (int)(tensorrt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)
    build_config = builder.create_builder_config()

    profile = builder.create_optimization_profile()
    profile.set_shape("input",
                        Dims([16, 3, 608, 608]),
                        Dims([16, 3, 608, 608]),
                        Dims([16, 3, 608, 608]),
    )
    build_config.add_optimization_profile(profile)

    parser = tensorrt.OnnxParser(network, IXRT_LOGGER)
    parser.parse_from_file(config.model)
    precision = tensorrt.BuilderFlag.INT8 if config.precision == "int8" else tensorrt.BuilderFlag.FP16
    # print("precision : ", precision)
    build_config.set_flag(precision)

    # set dynamic
    num_inputs = network.num_inputs
    for i in range(num_inputs):
        input_tensor = network.get_input(i)
        input_tensor.shape = Dims([16, 3, 608, 608])

    plan = builder.build_serialized_network(network, build_config)
    engine_file_path = config.engine
    with open(engine_file_path, "wb") as f:
        f.write(plan)
    print("Build dynamic shape engine done!")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--precision", type=str, choices=["float16", "int8", "float32"], default="int8",
            help="The precision of datatype")
    # engine args
    parser.add_argument("--engine", type=str, default=None)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)