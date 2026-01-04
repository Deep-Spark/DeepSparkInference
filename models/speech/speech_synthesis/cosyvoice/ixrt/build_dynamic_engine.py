# Copyright (c) 2024, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.

import os
import cv2
import argparse
import numpy as np

import torch
import tensorrt
from tensorrt import Dims

def main(config):
    IXRT_LOGGER = tensorrt.Logger(tensorrt.Logger.WARNING)
    builder = tensorrt.Builder(IXRT_LOGGER)
    EXPLICIT_BATCH = 1 << (int)(tensorrt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)
    build_config = builder.create_builder_config()
    profile = builder.create_optimization_profile()
    profile.set_shape("x", Dims([2, 80, 40]), Dims([2, 80, 500]), Dims([2, 80, 3000]))
    profile.set_shape("mask", Dims([2, 1, 4]), Dims([2, 1, 500]), Dims([2, 1, 3000]))
    profile.set_shape("mu", Dims([2, 80, 4]), Dims([2, 80, 500]), Dims([2, 80, 3000]))
    profile.set_shape("cond", Dims([2, 80, 4]), Dims([2, 80, 500]), Dims([2, 80, 3000]))

    tensor_dtype = tensorrt.DataType.FLOAT
    if config.precision == "float16":
        build_config.set_flag(tensorrt.BuilderFlag.FP16)
        tensor_dtype = tensorrt.DataType.HALF

    build_config.add_optimization_profile(profile)
    parser = tensorrt.OnnxParser(network, IXRT_LOGGER)
    parser.parse_from_file(config.model)

    # set input and output data type
    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        input_tensor.dtype = tensor_dtype
    for i in range(network.num_outputs):
        output_tensor = network.get_output(i)
        output_tensor.dtype = tensor_dtype

    plan = builder.build_serialized_network(network, build_config)
    engine_file_path = config.engine
    with open(engine_file_path, "wb") as f:
        f.write(plan)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--precision", type=str, choices=["float16", "float32"], default="float16",
            help="The precision of datatype")
    parser.add_argument("--engine", type=str, default=None)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)