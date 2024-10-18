#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
import sys
import time
import argparse
import numpy as np
from tqdm import tqdm
from copy import deepcopy

import torch
import onnxruntime

from common import check_target, get_dataloader, eval_batch

import tensorrt
from cuda import cuda, cudart


def parse_config():
    parser = argparse.ArgumentParser(description="IXRT lightweight openpose")
    parser.add_argument("--model_type", type=str, default="lightweight openpose", help="the model name")
    parser.add_argument("--test_mode", type=str, default="FPS", help="FPS MAP")
    parser.add_argument("--engine_file", type=str, help="engine file path")
    parser.add_argument("--datasets_dir", type=str, default="", help="ImageNet dir")
    parser.add_argument("--warm_up", type=int, default=-1, help="warm_up times")
    parser.add_argument("--bsz", type=int, default=16, help="test batch size")
    parser.add_argument("--imgh", type=int, default=256, help="inference size h")
    parser.add_argument("--imgw", type=int, default=456, help="inference size w")
    parser.add_argument("--use_async", action="store_true")
    parser.add_argument("--device", type=int, default=0, help="cuda device, i.e. 0 or 0,1,2,3,4")
    parser.add_argument("--fps_target", type=float, default=-1.0)
    parser.add_argument("--map_target", type=float, default=-1.0)
    parser.add_argument("--run_loop", type=int, default=-1)

    config = parser.parse_args()
    return config


def openpose_trtapi_ixrt(config):
    engine_file = config.engine_file
    datatype = tensorrt.DataType.FLOAT
    host_mem = tensorrt.IHostMemory
    logger = tensorrt.Logger(tensorrt.Logger.ERROR)
    with open(engine_file, "rb") as f, tensorrt.Runtime(logger) as runtime:
        runtime = tensorrt.Runtime(logger)
        assert runtime
        engine = runtime.deserialize_cuda_engine(f.read())
        assert engine
        context = engine.create_execution_context()
        assert context

    return engine, context


def setup_io_bindings(engine, context):
    # Setup I/O bindings
    inputs = []
    outputs = []
    allocations = []

    for i in range(engine.num_bindings):
        is_input = False
        if engine.binding_is_input(i):
            is_input = True
        name = engine.get_binding_name(i)
        dtype = engine.get_binding_dtype(i)
        shape = context.get_binding_shape(i)
        if is_input:
            batch_size = shape[0]
        size = np.dtype(tensorrt.nptype(dtype)).itemsize
        for s in shape:
            size *= s
        err, allocation = cudart.cudaMalloc(size)
        assert err == cudart.cudaError_t.cudaSuccess
        binding = {
            "index": i,
            "name": name,
            "dtype": np.dtype(tensorrt.nptype(dtype)),
            "shape": list(shape),
            "allocation": allocation,
            "nbytes": size,
        }
        allocations.append(allocation)
        if engine.binding_is_input(i):
            inputs.append(binding)
        else:
            outputs.append(binding)
    return inputs, outputs, allocations


def main(config):

    engine, context = openpose_trtapi_ixrt(config)
    inputs, outputs, allocations = setup_io_bindings(engine, context)

    output = np.zeros(outputs[0]["shape"], outputs[0]["dtype"])
    data_in = np.zeros(inputs[0]["shape"], inputs[0]["dtype"])

    err, = cuda.cuMemcpyHtoD(inputs[0]["allocation"], data_in, data_in.nbytes)
    assert(err == cuda.CUresult.CUDA_SUCCESS)

    # Warm up
    if config.warm_up > 0:
        print("\nWarm Start.")
        for i in range(config.warm_up):
            context.execute_v2(allocations)
        print("Warm Done.")

    if config.test_mode == "FPS":
        torch.cuda.synchronize()
        start_time = time.time()
        for i in range(config.run_loop):
            context.execute_v2(allocations)

        torch.cuda.synchronize()
        end_time = time.time()
        forward_time = end_time - start_time

        fps = config.run_loop * config.bsz / forward_time
        print(f"\nCheck FPS         Test : {fps}    Target:{config.fps_target}   State : {'Pass' if fps >= config.fps_target else 'Fail'}")


if __name__ == "__main__":
    config = parse_config()
    main(config)
