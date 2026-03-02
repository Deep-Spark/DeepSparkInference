#
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import gc
import os
import ctypes
from os.path import dirname, exists, join
import subprocess
import warnings
from collections import OrderedDict, defaultdict

import numpy as np
import tensorrt as trt
import torch
from cuda import cudart
# from polygraphy.backend.common import bytes_from_path
# from polygraphy.backend.trt import (
#     engine_from_bytes,
# )
import tensorrt

import onnx
from onnx import numpy_helper


TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

def load_ixrt_plugin(
    logger=trt.Logger(trt.Logger.WARNING), namespace="", dynamic_path=""
):
    if not dynamic_path:
        dynamic_path = join(dirname(trt.__file__), "lib", "libixrt_plugin.so")
    if not exists(dynamic_path):
        raise FileNotFoundError(
            f"The ixrt_plugin lib {dynamic_path} is not existed, please provided effective plugin path!"
        )
    ctypes.CDLL(dynamic_path, mode=ctypes.RTLD_GLOBAL)
    trt.init_libnvinfer_plugins(logger, namespace)
    print(f"Loaded plugin from {dynamic_path}")

load_ixrt_plugin()

# Map of TensorRT dtype -> torch dtype
trt_to_torch_dtype_dict = {
    trt.DataType.BOOL: torch.bool,
    trt.DataType.UINT8: torch.uint8,
    trt.DataType.INT8: torch.int8,
    trt.DataType.INT32: torch.int32,
    trt.DataType.INT64: torch.int64,
    trt.DataType.HALF: torch.float16,
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.BF16: torch.bfloat16,
}


def _CUASSERT(cuda_ret):
    err = cuda_ret[0]
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(
            f"CUDA ERROR: {err}, error code reference: https://nvidia.github.io/cuda-python/module/cudart.html#cuda.cudart.cudaError_t"
        )
    if len(cuda_ret) > 1:
        return cuda_ret[1]
    return None

class Engine:
    def __init__(
        self,
        engine_path,
    ):
        self.engine_path = engine_path
        self.runtime = None
        self.engine = None
        self.context = None
        self.buffers = OrderedDict()
        self.tensors = OrderedDict()
        self.cuda_graph_instance = None  # cuda graph
        self.logger=tensorrt.Logger(tensorrt.Logger.INFO)


    def __del__(self):
        del self.engine
        del self.context
        del self.buffers
        del self.tensors

    def refit(self, refit_weights, updated_weight_names):
        # Initialize refitter
        refitter = trt.Refitter(self.engine, TRT_LOGGER)
        refitted_weights = set()

        def refit_single_weight(trt_weight_name):
            # get weight from state dict
            trt_datatype = refitter.get_weights_prototype(trt_weight_name).dtype
            refit_weights[trt_weight_name] = refit_weights[trt_weight_name].to(trt_to_torch_dtype_dict[trt_datatype])

            # trt.Weight and trt.TensorLocation
            trt_wt_tensor = trt.Weights(
                trt_datatype, refit_weights[trt_weight_name].data_ptr(), torch.numel(refit_weights[trt_weight_name])
            )
            trt_wt_location = (
                trt.TensorLocation.DEVICE if refit_weights[trt_weight_name].is_cuda else trt.TensorLocation.HOST
            )

            # apply refit
            refitter.set_named_weights(trt_weight_name, trt_wt_tensor, trt_wt_location)
            refitted_weights.add(trt_weight_name)

        # iterate through all tensorrt refittable weights
        for trt_weight_name in refitter.get_all_weights():
            if trt_weight_name not in updated_weight_names:
                continue

            refit_single_weight(trt_weight_name)

        # iterate through missing weights required by tensorrt - addresses the case where lora_scale=0
        for trt_weight_name in refitter.get_missing_weights():
            refit_single_weight(trt_weight_name)

        if not refitter.refit_cuda_engine():
            print("Error: failed to refit new weights.")
            exit(0)

        print(f"[I] Total refitted weights {len(refitted_weights)}.")

    def load(self, weight_streaming=False, weight_streaming_budget_percentage=None):
        if self.engine is not None:
            print(f"[W]: Engine {self.engine_path} already loaded, skip reloading")
            return
        
        with open(self.engine_path, "rb") as f:
            self.runtime = tensorrt.Runtime(self.logger)
            assert self.runtime
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
            assert self.engine
        
        return self.engine

    def unload(self):
        if self.engine is not None:
            print(f"Unloading TensorRT engine: {self.engine_path}")
            del self.engine
            self.engine = None
            gc.collect()
        else:
            print(f"[W]: Unload an unloaded engine {self.engine_path}, skip unloading")

    def activate(self, device_memory=None):
        if device_memory:
            self.context = self.engine.create_execution_context_without_device_memory()
            self.context.device_memory = device_memory
        else:
            self.context = self.engine.create_execution_context()

    def reactivate(self, device_memory):
        assert self.context
        self.context.device_memory = device_memory

    def deactivate(self):
        del self.context
        self.context = None

    def allocate_buffers(self, shape_dict=None, device="cuda"):
        for binding in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(binding)
            if shape_dict and name in shape_dict:
                shape = shape_dict[name]
            else:
                shape = self.engine.get_tensor_shape(name)
                print(
                    f"[W]: {self.engine_path}: Could not find '{name}' in shape dict {shape_dict}.  Using shape {shape} inferred from the engine."
                )
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.context.set_input_shape(name, shape)
            dtype = trt_to_torch_dtype_dict[self.engine.get_tensor_dtype(name)]
            tensor = torch.empty(tuple(shape), dtype=dtype).to(device=device)
            self.tensors[name] = tensor

    def deallocate_buffers(self):
        for idx in range(self.engine.num_io_tensors):
            binding = self.engine[idx]
            del self.tensors[binding]

    def infer(self, feed_dict, stream):
        for name, buf in feed_dict.items():
            self.tensors[name].copy_(buf)

        for name, tensor in self.tensors.items():
            self.context.set_tensor_address(name, tensor.data_ptr())


        noerror = self.context.execute_async_v3(stream)
        if not noerror:
            raise ValueError(f"ERROR: inference of {self.engine_path} failed.")

        return self.tensors
