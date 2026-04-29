#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import glob
import json
import os
import sys
import time
import random
import ctypes
import numpy as np
from os.path import join, dirname, exists

from tqdm import tqdm

from utils import Dataset, get_confusion_matrix
import tensorrt
import cuda.cuda as cuda
import cuda.cudart as cudart

def load_ixrt_plugin(logger=tensorrt.Logger(tensorrt.Logger.INFO), namespace="", dynamic_path=""):
    if not dynamic_path:
        dynamic_path = join(dirname(tensorrt.__file__), "lib", "libixrt_plugin.so")
    if not exists(dynamic_path):
        raise FileNotFoundError(
            f"The ixrt_plugin lib {dynamic_path} is not existed, please provided effective plugin path!")
    ctypes.CDLL(dynamic_path)
    tensorrt.init_libnvinfer_plugins(logger, namespace)
    print(f"Loaded plugin from {dynamic_path}")

load_ixrt_plugin()

def create_engine_context(config):
    engine_path = config.engine_file
    datatype = tensorrt.DataType.FLOAT
    host_mem = tensorrt.IHostMemory
    logger = tensorrt.Logger(tensorrt.Logger.ERROR)
    with open(engine_path, "rb") as f, tensorrt.Runtime(logger) as runtime:
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

def check_target(inference, target):
    satisfied = False
    if inference > target:
        satisfied = True
    return satisfied


def test_mIoU_mAcc(dataset, config):
    
    confusion_matrix = np.zeros((config.num_classes, config.num_classes))
    
    host_mem = tensorrt.IHostMemory
    logger = tensorrt.Logger(tensorrt.Logger.ERROR)
    
    engine, context = create_engine_context(config)
    inputs, outputs, allocations = setup_io_bindings(engine, context)
    
    run_times = []
    
    for i, element in tqdm(enumerate(dataset), desc="Testing mIoU and mAcc"):
        start_time = time.time()
        img, label, pad_size, name = element
        img = np.ascontiguousarray(img.transpose((0,3,1,2)))
        b, c, h, w = img.shape
        
        output = np.zeros([b, 32, h, w], outputs[0]["dtype"])
        err, = cuda.cuMemcpyHtoD(inputs[0]["allocation"], img, img.nbytes)
        assert(err == cuda.CUresult.CUDA_SUCCESS)
        context.execute_v2(allocations)
        err, = cuda.cuMemcpyDtoH(output, outputs[0]["allocation"], outputs[0]["nbytes"])
        assert(err == cuda.CUresult.CUDA_SUCCESS)
        
        pred = output[:, :config.num_classes, :, :]
        # flip test
        if config.flip:
            flip_img = img.copy()[:, :, :, ::-1]

            err, = cuda.cuMemcpyHtoD(inputs[0]["allocation"], img, img.nbytes)
            assert(err == cuda.CUresult.CUDA_SUCCESS)
            context.execute_v2(allocations)
            err, = cuda.cuMemcpyDtoH(output, outputs[0]["allocation"], outputs[0]["nbytes"])
            assert(err == cuda.CUresult.CUDA_SUCCESS)
            flip_pred = output[:, :config.num_classes, :, :]
            
            pred += flip_pred
            out = np.exp(pred * 0.5)
        else:
            out = np.exp(pred)
            
        out = out.transpose((0,2,3,1))

        for j in range(b):
            confusion_matrix += get_confusion_matrix(
                label[j:j+1], 
                out[j:j+1], 
                pad_size[j], 
                config.num_classes, 
                config.ignore_label
                )
        
        end_time = time.time()
        run_times.append(end_time - start_time)
        
        num_imgs = i * config.bsz 
        if num_imgs % 100 == 0:
            print(f"[INFO] processing: {num_imgs} images")
            pos = confusion_matrix.sum(1)
            res = confusion_matrix.sum(0)
            tp = np.diag(confusion_matrix)
            IoU_array = (tp / np.maximum(1.0, pos + res - tp))
            mean_IoU = IoU_array.mean()
            print("[INFO] mIoU: %.4f" % (mean_IoU))
            
    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    pixel_acc = tp.sum() / pos.sum()
    mean_acc = (tp / np.maximum(1.0, pos)).mean()
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()

    # Calculate FPS
    run_times.remove(max(run_times))
    run_times.remove(min(run_times))
    avg_time = sum(run_times) / len(run_times)
    fps = 1. / avg_time 
    print(f"Executing Done, Time: {avg_time}, FPS: {fps}, mIoU: {mean_IoU}, mAcc: {mean_acc}")
    print(f"Class IoU:")
    print(f"{IoU_array}")
    metricResult = {"metricResult": {}}
    metricResult["metricResult"]["mIoU"] = round(mean_IoU, 3)
    metricResult["metricResult"]["mAcc"] = round(mean_acc, 3)
    print(metricResult)
    return mean_IoU, mean_acc
        

def test_fps(config, loop_count, dataset):
    
    host_mem = tensorrt.IHostMemory
    logger = tensorrt.Logger(tensorrt.Logger.ERROR)
    
    engine, context = create_engine_context(config)
    inputs, outputs, allocations = setup_io_bindings(engine, context)

    run_times = []
    
    if config.warm_up > 0:
        print("\nWarm Start.")
        for i in range(config.warm_up):
            context.execute_v2(allocations)
        print("Warm Done.")
    
    batch_data0 = dataset[0]
    for i in range(loop_count):
        img, label, pad_size, name = batch_data0
        b, h, w, c = img.shape
        output = np.zeros([b, 32, h, w], outputs[0]["dtype"])
        img = np.ascontiguousarray(img.transpose((0,3,1,2))) 
        err, = cuda.cuMemcpyHtoD(inputs[0]["allocation"], img, img.nbytes)
        assert(err == cuda.CUresult.CUDA_SUCCESS)
        start_time = time.time()
        context.execute_v2(allocations)
        end_time = time.time()
        err, = cuda.cuMemcpyDtoH(output, outputs[0]["allocation"], outputs[0]["nbytes"])
        assert(err == cuda.CUresult.CUDA_SUCCESS)

        temp_time = end_time - start_time
        fps = b / temp_time
        print(f"time: {temp_time}, fps: {fps}")
        run_times.append(temp_time)
            
    # Calculate FPS
    run_times.remove(max(run_times))
    run_times.remove(min(run_times))

    avg_time = sum(run_times) / len(run_times)
    fps = b / avg_time 
    print(f"Executing {loop_count} done, Time: {avg_time}, FPS: {fps}")
    metricResult = {"metricResult": {}}
    metricResult["metricResult"]["FPS"] = round(fps, 3)
    print(metricResult)
    return fps


def main(config):

    num_samples = 1
    bsz = config.bsz
    if config.loop_count > 0:
        num_samples = bsz * config.loop_count
    num_batch = (num_samples + bsz - 1) // bsz
    
    dataset = Dataset(
            root=config.dataset_dir, 
            list_path=config.list_path, 
            batch_size=config.bsz,
            ignore_label=255
        )
    
    if config.test_mode == "MIOU":
        mIoU, mAcc = test_mIoU_mAcc(dataset, config)       
        status_mIoU_mAcc = check_target(mIoU, config.target_mIoU) and check_target(mAcc, config.target_mAcc)
        sys.exit(int(not (status_mIoU_mAcc)))
    
    elif config.test_mode == "FPS":
        # Warm up
        fps = test_fps(config, config.loop_count, dataset)    
        status_fps = check_target(fps, config.target_fps)
        sys.exit(int(not (status_fps)))
        
    
def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        type=str,
        default="DDRNET",
        help="The semantic segmentation(ddrnet)",
    )
    parser.add_argument("--engine_file", type=str, help="engine file path")
    parser.add_argument("--test_mode", type=str, default="MIOU", help="FPS MIOU")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/root/data/datasets",
        help="The directory of dataset(cityscapes)",
    )
    parser.add_argument(
        "--list_path",
        type=str,
        default="/root/data/datasets/cityscapes/val.lst",
        help="The val name list of dataset(cityscapes)",
    )
    parser.add_argument("--warm_up", type=int, default=5, help="warm_up count")
    parser.add_argument("--flip", action='store_true', help="Flip test")
    parser.add_argument("--bsz", type=int, default=4, help="batch size")
    parser.add_argument("--num_classes", type=int, default=19, help="the category of dataset")
    parser.add_argument("--ignore_label", type=int, default=255, help="the category of not used in calculate confusion matrix")
    parser.add_argument("--imgsz_h", type=int, default=1024, help="inference size h")
    parser.add_argument("--imgsz_w", type=int, default=2048, help="inference size w")
    parser.add_argument("--pred_dir", type=str, default=".", help="pred save json dirs")
    parser.add_argument("--target_fps", type=float, default=-1.0)
    parser.add_argument("--target_mIoU", type=float, default=-1.0)
    parser.add_argument("--target_mAcc", type=float, default=-1.0)
    parser.add_argument("--loop_count", type=int, default=12)
    parser.add_argument(
        "--device", type=int, default=0, help="cuda device, i.e. 0 or 0,1,2,3,4"
    )

    config = parser.parse_args()
    return config


if __name__ == "__main__":
    config = parse_config()
    main(config)
