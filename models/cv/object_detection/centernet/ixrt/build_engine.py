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
    profile.set_shape("input", Dims([32, 3, 672, 672]), Dims([32, 3, 672, 672]), Dims([32, 3, 672, 672]))
    build_config.add_optimization_profile(profile)
    parser = tensorrt.OnnxParser(network, IXRT_LOGGER)
    parser.parse_from_file(config.model)

    precision = tensorrt.BuilderFlag.INT8 if config.precision == "int8" else tensorrt.BuilderFlag.FP16
    # print("precision : ", precision)
    build_config.set_flag(precision)
    num_inputs = network.num_inputs

    for i in range(num_inputs):
        input_tensor = network.get_input(i)
        input_tensor.shape = Dims([32, 3, 672, 672])

    plan = builder.build_serialized_network(network, build_config)
    engine_file_path = config.engine
    with open(engine_file_path, "wb") as f:
        f.write(plan)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--precision", type=str, choices=["float16", "int8", "float32"], default="int8",
            help="The precision of datatype")
    parser.add_argument("--engine", type=str, default=None)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)