import os
import cv2
import glob
import torch
import tensorrt
import numpy as np
from cuda import cuda, cudart

from torch.utils.data import DataLoader, SubsetRandomSampler, SequentialSampler
from torchvision import datasets, transforms

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
        print(f"binding {i}, name : {name}  dtype : {np.dtype(tensorrt.nptype(dtype))}  shape : {list(shape)}")
        allocations.append(allocation)
        if engine.binding_is_input(i):
            inputs.append(binding)
        else:
            outputs.append(binding)
    return inputs, outputs, allocations

def fixed_image_standardization(image_tensor):
    processed_tensor = (image_tensor - 127.5) / 128.0
    return processed_tensor

def collate_pil(x): 
    out_x, out_y = [], [] 
    for xx, yy in x: 
        out_x.append(xx) 
        out_y.append(yy) 
    return out_x, out_y 

def getdataloader(datasets_dir, step=20, batch_size=64, image_size=160):
    orig_img_ds = datasets.ImageFolder(datasets_dir + 'lfw', transform=None)
    orig_img_ds.samples = [
        (p, p)
        for p, _ in orig_img_ds.samples
    ]
    loader = DataLoader(
        orig_img_ds,
        num_workers=16,
        batch_size=batch_size,
        collate_fn=collate_pil
    )
    crop_paths = []
    box_probs = []
    for i, (x, b_paths) in enumerate(loader):
        crops = [p for p in b_paths]
        crop_paths.extend(crops)
        # print('\rBatch {} of {}'.format(i + 1, len(loader)), end='')

    trans = transforms.Compose([
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization
    ])

    dataset = datasets.ImageFolder(datasets_dir + 'lfw', transform=trans)
    embed_loader = DataLoader(
        dataset,
        num_workers=16,
        batch_size=batch_size,
        sampler=SequentialSampler(dataset)
    )

    return embed_loader, crop_paths
