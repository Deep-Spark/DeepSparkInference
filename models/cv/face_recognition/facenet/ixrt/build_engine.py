import os
import cv2
import argparse
import numpy as np

import torch
import tensorrt

import onnx
from onnx import helper
from onnx import TensorProto,numpy_helper
from load_ixrt_plugin import load_ixrt_plugin
load_ixrt_plugin()

def add_facenet_norm(onnx_model):
    norm = helper.make_node('FacenetNorm_IxRT', inputs=['1189'] , outputs=['1190'], name='facenet_norm_1', size=512)
    
    onnx_model = onnx.load(onnx_model)
    graph = onnx_model.graph
    nodes = graph.node
    graph.node.append(norm)
    output = onnx.helper.make_tensor_value_info('1190', TensorProto.FLOAT, [64, 512, 1, 1])
    graph = onnx.helper.make_graph(
        graph.node,
        "facenet model",
        graph.input,
        [output],
        graph.initializer
    )
    info_model = onnx.helper.make_model(graph, producer_name="facenet")
    info_model.opset_import[0].version = 11
    onnx.save(info_model, "tmp4.onnx")

def main(config):
    IXRT_LOGGER = tensorrt.Logger(tensorrt.Logger.WARNING)
    builder = tensorrt.Builder(IXRT_LOGGER)
    EXPLICIT_BATCH = 1 << (int)(tensorrt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)
    build_config = builder.create_builder_config()
    print("start prepare...")
    add_facenet_norm(config.model)
    parser = tensorrt.OnnxParser(network, IXRT_LOGGER)
    parser.parse_from_file("tmp4.onnx")

    precision = tensorrt.BuilderFlag.INT8 if config.precision == "int8" else tensorrt.BuilderFlag.FP16
    # print("precision : ", precision)
    build_config.set_flag(precision)

    plan = builder.build_serialized_network(network, build_config)
    engine_file_path = config.engine
    with open(engine_file_path, "wb") as f:
        f.write(plan)
    os.remove("tmp4.onnx")

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