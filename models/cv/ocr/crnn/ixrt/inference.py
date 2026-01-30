#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import time
from tqdm import tqdm

import numpy as np
from cuda import cuda, cudart
import torch
import tensorrt

import onnx
import onnxruntime
from sklearn.metrics.pairwise import cosine_similarity

from common import create_engine_context, get_io_bindings
from load_ixrt_plugin import load_ixrt_plugin
load_ixrt_plugin()

np.random.seed(0)

def onnxruntime_infer(onnx_path, input_data):
    # model = onnx.load(onnx_path)
    session = onnxruntime.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print("input_name:", input_name)
    print("outputs:", output_name)
    onnx_outs = session.run([output_name], {input_name: input_data})

    print(f"onnx output shape, {output_name} : {onnx_outs[0].shape}")
    return onnx_outs[0]

def compare(onnxruntime_output, ixrt_output):
    onnxruntime_output = onnxruntime_output.flatten()
    ixrt_output = ixrt_output.flatten()

    diff = np.abs(onnxruntime_output - ixrt_output)
    max_diff = np.max(diff)
    cos_sim = cosine_similarity(onnxruntime_output.reshape(1,-1),ixrt_output.reshape(1,-1))

    print("max diff : ", max_diff)
    print("cos sim : ", cos_sim)
    if cos_sim > 0.98:
        print("pass!")
    print("Onnxruntime output", onnxruntime_output[:30])
    print("\nIxRT output", ixrt_output[:30])
    metricResult = {"metricResult": {}}
    metricResult["metricResult"]["max_diff"] = max_diff
    metricResult["metricResult"]["cos_sim"] = cos_sim[0][0]
    print(metricResult)

def main(config):
    host_mem = tensorrt.IHostMemory
    logger = tensorrt.Logger(tensorrt.Logger.ERROR)

    # Load Engine && I/O bindings
    engine, context = create_engine_context(config.engine_file, logger)
    inputs, outputs, allocations = get_io_bindings(engine)

    # Warm up
    if config.warm_up > 0:
        print("\nWarm Start.")
        for i in range(config.warm_up):
            context.execute_v2(allocations)
        print("Warm Done.")

    # Inference
    if config.test_mode == "FPS":
        torch.cuda.synchronize()
        start_time = time.time()

        for i in range(config.loop_count):
            context.execute_v2(allocations)

        torch.cuda.synchronize()
        end_time = time.time()
        forward_time = end_time - start_time

        fps = config.loop_count * config.bsz / forward_time

        print("FPS : ", fps)
        print(f"Performance Check : Test {fps} >= target {config.fps_target}")
        if fps >= config.fps_target:
            print("pass!")
            exit()
        else:
            print("failed!")
            exit(1)

    elif config.test_mode == "ACC":
        
        ## Prepare the output data
        output = np.zeros(outputs[0]["shape"], outputs[0]["dtype"])
        print(f"output shape : {output.shape} output type : {output.dtype}")
        
        ## Prepare the input data
        batch_data = np.random.rand(config.bsz, 3, 32, 1920).astype(inputs[0]["dtype"])
        # batch_data = np.ones((config.bsz, 3, 32, 1920),dtype="float32") - 0.8
        batch_data = np.ascontiguousarray(batch_data)
        
        err, = cuda.cuMemcpyHtoD(inputs[0]["allocation"], batch_data, batch_data.nbytes)
        assert(err == cuda.CUresult.CUDA_SUCCESS)
        context.execute_v2(allocations)
        err, = cuda.cuMemcpyDtoH(output, outputs[0]["allocation"], outputs[0]["nbytes"])
        assert(err == cuda.CUresult.CUDA_SUCCESS)
        
        # compare onnxruntime and ixrt results
        for bs in range(config.bsz):
            batch_data_ = batch_data[bs, :, :, :]
            batch_data_ = np.expand_dims(batch_data_, axis=0)
            onnx_output = onnxruntime_infer(config.ort_onnx, batch_data_)
            compare(onnx_output, output[bs,:])

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_mode", type=str, default="FPS", help="FPS MAP")
    parser.add_argument(
        "--engine_file",
        type=str,
        help="engine file path"
    )
    parser.add_argument(
        "--datasets_dir",
        type=str,
        default="",
        help="ImageNet dir",
    )
    parser.add_argument("--warm_up", type=int, default=-1, help="warm_up times")
    parser.add_argument("--bsz", type=int, default=32, help="test batch size")
    parser.add_argument(
        "--imgsz",
        "--img",
        "--img-size",
        type=int,
        default=224,
        help="inference size h,w",
    )
    parser.add_argument("--use_async", action="store_true")
    parser.add_argument(
        "--device", type=int, default=0, help="cuda device, i.e. 0 or 0,1,2,3,4"
    )
    parser.add_argument("--fps_target", type=float, default=-1.0)
    parser.add_argument("--acc_target", type=float, default=-1.0)
    parser.add_argument("--loop_count", type=int, default=-1)
    parser.add_argument("--onnx_path", type=str)
    parser.add_argument("--ort_onnx", type=str)

    config = parser.parse_args()
    return config

if __name__ == "__main__":
    config = parse_config()
    main(config)
