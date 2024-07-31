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

"""
Build Engine From FusionPlugin Onnx.
"""

import os
import ctypes
import json
import onnx
import logging
import argparse

import tensorrt
import tensorrt as trt
from tensorrt import Dims


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
def load_ixrt_plugin(logger=trt.Logger(trt.Logger.WARNING), namespace="", dynamic_path=""):
    if not dynamic_path:
        dynamic_path = os.path.join(os.path.dirname(trt.__file__), "lib", "libixrt_plugin.so")
    if not os.path.exists(dynamic_path):
        raise FileNotFoundError(
            f"The ixrt_plugin lib {dynamic_path} is not existed, please provided effective plugin path!"
        )
    ctypes.CDLL(dynamic_path, mode=ctypes.RTLD_GLOBAL)
    trt.init_libnvinfer_plugins(logger, namespace)
    print(f"Loaded plugin from {dynamic_path}")

load_ixrt_plugin()



def parse_args():
    parser = argparse.ArgumentParser(description="build tensorrt engine of conformer.", usage="")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="conformer",
    )
    parser.add_argument(
        "--onnx_path",
        type=str,
        required=True,
        help="onnx_path path to save",
    )
    parser.add_argument(
        "--engine_path",
        type=str,
        required=True,
        help="engine path to save",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        required=True,
    )
    args = parser.parse_args()
    return args

args = parse_args()
MaxBSZ = args.max_batch_size
MaxSeqLen = args.max_seq_len


def build_engine_trtapi_dynamicshape(args):
    onnx_model = args.onnx_path
    assert os.path.isfile(onnx_model), f"The onnx model{onnx_model} must be existed!"
    IXRT_LOGGER = tensorrt.Logger(tensorrt.Logger.WARNING)
    builder = tensorrt.Builder(IXRT_LOGGER)
    EXPLICIT_BATCH = 1 << (int)(tensorrt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)
    build_config = builder.create_builder_config()

    profile = builder.create_optimization_profile()
    profile.set_shape("input", Dims([MaxBSZ, 100, 80]), Dims([MaxBSZ, 1000, 80]), Dims([MaxBSZ, 1500, 80]))
    profile.set_shape("mask", Dims([MaxBSZ, 1, 25]), Dims([MaxBSZ, 1, 250]), Dims([MaxBSZ, 1, 374]))
    profile.set_shape("pos_emb", Dims([1, 25, 256]), Dims([1, 250, 256]), Dims([1, 374, 256]))
    build_config.add_optimization_profile(profile)

    parser = tensorrt.OnnxParser(network, IXRT_LOGGER)
    parser.parse_from_file(onnx_model)
    build_config.set_flag(tensorrt.BuilderFlag.FP16)

    # set dynamic
    # input
    input_tensor = network.get_input(0)
    input_tensor.shape = Dims([MaxBSZ, -1, 80])
    # mask
    mask_tensor = network.get_input(1)
    mask_tensor.shape = Dims([MaxBSZ, 1, -1])
    # pos_emb
    pos_emb_tensor = network.get_input(2)
    pos_emb_tensor.shape = Dims([1, -1, 256])

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
    build_engine_trtapi_dynamicshape(args)
    # build_engine_trtapi_staticshape(args)
