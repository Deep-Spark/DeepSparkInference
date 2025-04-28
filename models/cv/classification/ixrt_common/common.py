import os
import cv2
import glob
import torch
import tensorrt
import numpy as np
from cuda import cuda, cudart

def eval_batch(batch_score, batch_label):
    batch_score = torch.tensor(torch.from_numpy(batch_score), dtype=torch.float32)
    values, indices = batch_score.topk(5)
    top1, top5 = 0, 0
    for idx, label in enumerate(batch_label):

        if label == indices[idx][0]:
            top1 += 1
        if label in indices[idx]:
            top5 += 1
    return top1, top5

def create_engine_context(engine_path, logger):
    with open(engine_path, "rb") as f:
        runtime = tensorrt.Runtime(logger)
        assert runtime
        engine = runtime.deserialize_cuda_engine(f.read())
        assert engine
        context = engine.create_execution_context()
        assert context

    return engine, context

def _get_engine_io_bindings(engine):
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
        print(f"binding {i}, name : {name}  dtype : {np.dtype(tensorrt.nptype(dtype))}  shape : {list(shape)}")
        allocations.append(allocation)
        if engine.binding_is_input(i):
            inputs.append(binding)
        else:
            outputs.append(binding)
    return inputs, outputs, allocations

def _convert_dtype_ort_to_np(type):
    type = type.replace("tensor", "")
    type = type.replace("(", "")
    type = type.replace(")", "")
    if type == "float":
        type = "float32"
    return np.dtype(type)

def _get_bytes_of_tensor(shape, type: np.dtype):
    size = type.itemsize
    for s in shape:
        size *= s
    return size
def _alloc_gpu_tensor(shape, dtype):
    size = _get_bytes_of_tensor(shape, dtype)
    err, allocation = cudart.cudaMalloc(size)
    assert err == cudart.cudaError_t.cudaSuccess
    return allocation

def _alloc_onnx_io_binding(io, index):
    type = _convert_dtype_ort_to_np(io.type)
    binding = {
        "index": index,
        "name": io.name,
        "dtype": type,
        "shape": io.shape,
        "allocation": None,
        "nbytes": _get_bytes_of_tensor(io.shape, type),
    }
    return binding
def _get_onnx_io_bindings(ort_session):
    # Setup I/O bindings
    inputs = []
    outputs = []
    allocations = []

    index = 0
    for input in ort_session.get_inputs():
        binding = _alloc_onnx_io_binding(input, index)
        index+=1
        inputs.append(binding)
        allocations.append(binding)
    for output in ort_session.get_outputs():
        binding = _alloc_onnx_io_binding(output, index)
        index+=1
        outputs.append(binding)
        allocations.append(binding)
    return inputs, outputs, allocations

def get_io_bindings(engine):
    if isinstance(engine, tensorrt.ICudaEngine):
        return _get_engine_io_bindings(engine)
    else:
        return _get_onnx_io_bindings(engine)
