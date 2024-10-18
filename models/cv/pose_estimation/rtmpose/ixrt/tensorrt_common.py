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

import cuda.cudart as cudart
import numpy as np
import tensorrt



def create_engine_from_onnx(onnx_file, engine_file):
    IXRT_LOGGER = tensorrt.Logger(tensorrt.Logger.ERROR)
    builder = tensorrt.Builder(IXRT_LOGGER)
    EXPLICIT_BATCH = 1 << (int)(
        tensorrt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH
    )
    network = builder.create_network(EXPLICIT_BATCH)
    build_config = builder.create_builder_config()
    parser = tensorrt.OnnxParser(network, IXRT_LOGGER)
    parser.parse_from_file(onnx_file)
    precision = tensorrt.BuilderFlag.FP16
    build_config.set_flag(precision)
    plan = builder.build_serialized_network(network, build_config)
    with open(engine_file, "wb") as f:
        f.write(plan)
        
def create_context(engine_file):
    
    host_mem = tensorrt.IHostMemory
    logger = tensorrt.Logger(tensorrt.Logger.ERROR)
    engine, context = create_engine_context(engine_file, logger)
    
    return engine, context
            
        
        
def get_ixrt_output(engine, context, input_x):
    
    inputs, outputs, allocations = get_io_bindings(engine)
    
    input_data = input_x.astype(inputs[0]["dtype"])
    input_data = np.ascontiguousarray(input_data)
    assert inputs[0]["nbytes"] == input_data.nbytes
    (err,) = cudart.cudaMemcpy(
        inputs[0]["allocation"],
        input_data,
        inputs[0]["nbytes"],
        cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
    )
    assert err == cudart.cudaError_t.cudaSuccess
    
    output0 = np.zeros(outputs[0]["shape"], outputs[0]["dtype"])
    output1 = np.zeros(outputs[1]["shape"], outputs[1]["dtype"])
    
    
    context.execute_v2(allocations)
    assert outputs[0]["nbytes"] == output0.nbytes
    (err,) = cudart.cudaMemcpy(
        output0,
        outputs[0]["allocation"],
        outputs[0]["nbytes"],
        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
    )
    assert err == cudart.cudaError_t.cudaSuccess
    
    
    assert outputs[1]["nbytes"] == output1.nbytes
    (err,) = cudart.cudaMemcpy(
        output1,
        outputs[1]["allocation"],
        outputs[1]["nbytes"],
        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
    )
    assert err == cudart.cudaError_t.cudaSuccess
    # Free
    for alloc in allocations:
        (err,) = cudart.cudaFree(alloc)
        assert err == cudart.cudaError_t.cudaSuccess
    return output0,output1        


def create_engine_context(engine_path, logger):
    with open(engine_path, "rb") as f:
        runtime = tensorrt.Runtime(logger)
        assert runtime
        engine = runtime.deserialize_cuda_engine(f.read())
        assert engine
        context = engine.create_execution_context()
        assert context

    return engine, context


def get_io_bindings(engine):
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
        shape = engine.get_binding_shape(i)
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
        print(
            f"binding {i}, name : {name}  dtype : {np.dtype(tensorrt.nptype(dtype))}  shape : {list(shape)}"
        )
        allocations.append(allocation)
        if engine.binding_is_input(i):
            inputs.append(binding)
        else:
            outputs.append(binding)
    return inputs, outputs, allocations


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
