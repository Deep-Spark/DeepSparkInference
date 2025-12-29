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
import time
import argparse
import torch
import numpy as np
from tqdm import tqdm
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt

from load_ixrt_plugin import load_ixrt_plugin
load_ixrt_plugin()

from utils import COCO2017Dataset, COCO2017Evaluator

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--engine",
                        type=str,
                        required=True,
                        help="igie engine path.")

    parser.add_argument("--batchsize",
                        type=int,
                        required=True,
                        help="inference batch size.")

    parser.add_argument("--datasets",
                        type=str,
                        required=True,
                        help="datasets path.")

    parser.add_argument("--warmup",
                        type=int,
                        default=5,
                        help="number of warmup before test.")

    parser.add_argument("--num_workers",
                        type=int,
                        default=16,
                        help="number of workers used in pytorch dataloader.")

    parser.add_argument("--acc_target",
                        type=float,
                        default=None,
                        help="Model inference Accuracy target.")

    parser.add_argument("--fps_target",
                        type=float,
                        default=None,
                        help="Model inference FPS target.")

    parser.add_argument("--conf",
                        type=float,
                        default=0.001,
                        help="confidence threshold.")

    parser.add_argument("--iou",
                        type=float,
                        default=0.65,
                        help="iou threshold.")

    parser.add_argument("--perf_only",
                        type=bool,
                        default=False,
                        help="Run performance test only")
    parser.add_argument("--loop_count",
                        type=int,
                        default=-1,
                        help="loop count")

    args = parser.parse_args()

    return args

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
        size = np.dtype(trt.nptype(dtype)).itemsize
        for s in shape:
            size *= s
        allocation = cuda.mem_alloc(size)
        binding = {
            "index": i,
            "name": name,
            "dtype": np.dtype(trt.nptype(dtype)),
            "shape": list(shape),
            "allocation": allocation,
        }
        print(f"binding {i}, name : {name}  dtype : {np.dtype(trt.nptype(dtype))}  shape : {list(shape)}")
        allocations.append(allocation)
        if engine.binding_is_input(i):
            inputs.append(binding)
        else:
            outputs.append(binding)
    return inputs, outputs, allocations

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
        allocation = cuda.mem_alloc(size)
        binding = {
            "index": i,
            "name": name,
            "dtype": np.dtype(tensorrt.nptype(dtype)),
            "shape": list(shape),
            "allocation": allocation,
        }
        print(f"binding {i}, name : {name}  dtype : {np.dtype(tensorrt.nptype(dtype))}  shape : {list(shape)}")
        allocations.append(allocation)
        if engine.binding_is_input(i):
            inputs.append(binding)
        else:
            outputs.append(binding)
    return inputs, outputs, allocations

def get_dataloader(data_path, label_path, batch_size, num_workers):

    dataset = COCO2017Dataset(data_path, label_path, image_size=(640, 640))

    dataloader = torch.utils.data.DataLoader(dataset,
                                    batch_size=batch_size,
                                    drop_last=False,
                                    num_workers=num_workers,
                                    collate_fn=dataset.collate_fn)
    return dataloader

def main():
    args = parse_args()

    batch_size = args.batchsize
    data_path = os.path.join(args.datasets, "val2017")
    label_path = os.path.join(args.datasets, "annotations", "instances_val2017.json")


    host_mem = tensorrt.IHostMemory
    logger = tensorrt.Logger(tensorrt.Logger.ERROR)

    # Load Engine
    engine, context = create_engine_context(args.engine, logger)
    inputs, outputs, allocations = get_io_bindings(engine)

    # Warm up
    print("\nWarm Start.")
    for i in range(args.warmup):
        context.execute_v2(allocations)
    print("Warm Done.")

    output_np = np.zeros(outputs[0]["shape"], outputs[0]["dtype"])
    # just run perf test
    if args.perf_only:
        start_time = time.time()
        for i in range(args.loop_count):
            context.execute_v2(allocations)
        end_time = time.time()
        forward_time = end_time - start_time
        fps = args.loop_count / forward_time * batch_size
        print("FPS : ", fps)
    else:
        # get dataloader
        dataloader = get_dataloader(data_path, label_path, batch_size, args.num_workers)

        # get evaluator
        evaluator = COCO2017Evaluator(label_path=label_path,
                                    conf_thres=args.conf,
                                    iou_thres=args.iou,
                                    image_size=640)
        start_time = time.time()
        for all_inputs in tqdm(dataloader):
            image = all_inputs[0]
            pad_batch = len(image) != batch_size
            if pad_batch:
                origin_size = len(image)
                image = np.resize(image, (batch_size, *image.shape[1:]))

            cuda.memcpy_htod(inputs[0]["allocation"], image)
            context.execute_v2(allocations)

            cuda.memcpy_dtoh(output_np, outputs[0]["allocation"])
            # print("output_np")
            # print(output_np)

            if pad_batch:
                output_np = output_np[:origin_size]

            evaluator.evaluate(output_np, all_inputs)
        end_time = time.time()
        end2end_time = end_time - start_time
        print(F"E2E time : {end2end_time:.3f} seconds")

        evaluator.summary()

if __name__ == "__main__":
    main()