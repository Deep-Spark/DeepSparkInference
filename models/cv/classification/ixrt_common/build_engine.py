import os
import cv2
import argparse
import numpy as np

import torch
import tensorrt
import ixrt

TRT_LOGGER = tensorrt.Logger(tensorrt.Logger.WARNING)
from load_ixrt_plugin import load_ixrt_plugin
load_ixrt_plugin(TRT_LOGGER)

def main(config):
    if config.silent:
        action = tensorrt.Logger.WARNING
    else:
        action = tensorrt.Logger.ERROR
    IXRT_LOGGER = tensorrt.Logger(action)
    builder = tensorrt.Builder(IXRT_LOGGER)
    EXPLICIT_BATCH = 1 << (int)(tensorrt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)
    build_config = builder.create_builder_config()
    parser = tensorrt.OnnxParser(network, IXRT_LOGGER)
    if not parser.parse_from_file(config.model):
        raise Exception(f"Failed to parse {config.model}, please check detailed debug info")

    precision = tensorrt.BuilderFlag.INT8 if config.precision == "int8" else tensorrt.BuilderFlag.FP16
    # print("precision : ", precision)
    build_config.set_flag(precision)

    # due to fp16 of elementwise div of swin_v2 will exceed the range of f16 representation， so set fp32.
    if ("swin_v2_s_model_sim" in config.model) and (config.precision == "float16"):
        build_config.set_flag(ixrt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        float32_set = {"/features/features.3/features.3.0/attn/Div_4",
                    "/features/features.3/features.3.1/attn/Div_4",
                    "/features/features.5/features.5.0/attn/Div_4",
                    "/features/features.5/features.5.1/attn/Div_4",
                    "/features/features.5/features.5.2/attn/Div_4",
                    "/features/features.5/features.5.3/attn/Div_4",
                    "/features/features.5/features.5.4/attn/Div_4",
                    "/features/features.5/features.5.5/attn/Div_4",
                    "/features/features.5/features.5.6/attn/Div_4",
                    "/features/features.5/features.5.7/attn/Div_4",
                    "/features/features.5/features.5.8/attn/Div_4",
                    "/features/features.5/features.5.9/attn/Div_4",
                    "/features/features.5/features.5.10/attn/Div_4",
                    "/features/features.5/features.5.11/attn/Div_4",
                    "/features/features.5/features.5.12/attn/Div_4",
                    "/features/features.5/features.5.13/attn/Div_4",
                    "/features/features.5/features.5.14/attn/Div_4",
                    "/features/features.5/features.5.15/attn/Div_4",
                    "/features/features.5/features.5.16/attn/Div_4",
                    "/features/features.5/features.5.17/attn/Div_4",
                    "/features/features.7/features.7.0/attn/Div_4",
                    "/features/features.7/features.7.1/attn/Div_4",}
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            if layer.name in float32_set:
                layer.precision = ixrt.float32
    
    plan = builder.build_serialized_network(network, build_config)
    if not plan:
        raise Exception("Failed to build engine, please check detailed debug info")
    engine_file_path = config.engine
    with open(engine_file_path, "wb") as f:
        f.write(plan)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--precision", type=str, choices=["float16", "int8", "float32"], default="int8",
            help="The precision of datatype")
    parser.add_argument("--engine", type=str, default=None)
    parser.add_argument("--silent", action="store_true")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
