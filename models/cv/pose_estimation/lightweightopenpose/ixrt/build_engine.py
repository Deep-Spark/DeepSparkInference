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
import json
import onnx
import logging
import argparse

import tensorrt
from tensorrt import Dims


def parse_config():
    parser = argparse.ArgumentParser(description="Build tensorrt engine of lightweight openpose", usage="")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="model name lightweight openpose",
    )
    parser.add_argument(
        "--onnx_path",
        type=str,
        required=True,
        help="The onnx path",
    )
    parser.add_argument(
        "--engine_path",
        type=str,
        required=True,
        help="engine path to save",
    )
    parser.add_argument(
        "--engine_path_dynamicshape",
        type=str,
        required=True,
        help="engine path to save(dynamic)",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="cuda device, i.e. 0 or 0,1,2,3,4"
    )
    config = parser.parse_args()
    return config


def build_engine_trtapi(config):
    onnx_model = config.onnx_path
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
    with open(config.engine_path, "wb") as f:
        f.write(plan)

    print("Build fixed shape engine done!")


def build_engine_trtapi_dynamicshape(config):
    onnx_model = config.onnx_path
    assert os.path.isfile(onnx_model), f"The onnx model{onnx_model} must be existed!"
    IXRT_LOGGER = tensorrt.Logger(tensorrt.Logger.WARNING)
    builder = tensorrt.Builder(IXRT_LOGGER)
    EXPLICIT_BATCH = 1 << (int)(tensorrt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)
    build_config = builder.create_builder_config()

    profile = builder.create_optimization_profile()
    profile.set_shape(
        "data", Dims([1, 3, 340, 340]), Dims([1, 3, 368, 744]), Dims([1, 3, 380, 1488])
    )
    build_config.add_optimization_profile(profile)

    parser = tensorrt.OnnxParser(network, IXRT_LOGGER)

    parser.parse_from_file(onnx_model)
    build_config.set_flag(tensorrt.BuilderFlag.FP16)

    # set dynamic
    input_tensor = network.get_input(0)
    input_tensor.shape = Dims([-1, 3, -1, -1])

    plan = builder.build_serialized_network(network, build_config)
    with open(config.engine_path_dynamicshape, "wb") as f:
        f.write(plan)

    print("Build dynamic shape engine done!")


if __name__ == "__main__":
    config = parse_config()
    build_engine_trtapi(config)
    build_engine_trtapi_dynamicshape(config)
