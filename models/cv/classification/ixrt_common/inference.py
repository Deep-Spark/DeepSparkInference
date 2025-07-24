#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import time
from tqdm import tqdm

import cv2
import numpy as np
from cuda import cuda, cudart
import torch
import tensorrt

from calibration_dataset import getdataloader, getmobilenetv1dataloader, getclipdataloader
from common import eval_batch, create_engine_context, get_io_bindings

TRT_LOGGER = tensorrt.Logger(tensorrt.Logger.WARNING)
from load_ixrt_plugin import load_ixrt_plugin
load_ixrt_plugin(TRT_LOGGER)

class ModelRunner:
    def __init__(self, model_path, logger):
        self.model_path = model_path
        self.logger = logger

        if model_path.endswith(".onnx"):
            self.backend = "onnxruntime"
        elif  model_path.endswith(".engine"):
            self.backend = "ixrt"
        else:
            raise Exception("No supported backend for executing ", model_path, "only support engine/onnx format")

        if self.is_ixrt_backend():
            self.init_ixrt()
        elif self.is_ort_backend():
            self.init_onnxruntime()
        else:
            raise Exception("No supported backend for", self.backend)
    def is_ixrt_backend(self):
        return self.backend == "ixrt"

    def is_ort_backend(self):
        return self.backend == "onnxruntime"
    def init_ixrt(self):
        self.engine, self.context = create_engine_context(self.model_path, self.logger)
        self.inputs, self.outputs, self.allocations = get_io_bindings(self.engine)

    def init_onnxruntime(self):
        import onnxruntime, onnx
        raw_onnx = onnx.load(self.model_path)
        self.ort_session = onnxruntime.InferenceSession(
            raw_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        self.inputs, self.outputs, self.allocations = get_io_bindings(self.ort_session)

    def run(self):
        if self.is_ixrt_backend():
            self.run_ixrt()
        elif self.is_ort_backend():
            self.run_onnxruntime()
        else:
            raise Exception("No supported backend for", self.backend)

    def run_onnxruntime(self):
        input_buffers = {}
        for input in self.inputs:
            input_buffers[input["name"]] = input["allocation"]

        output_names = [output["name"] for output in self.outputs]
        ort_outs = self.ort_session.run(output_names, input_buffers)

        for i in range(len(output_names)):
            self.outputs[i]["allocation"] = ort_outs[i]
    def run_ixrt(self):
        self.context.execute_v2(self.allocations)

class ClassificationRunner(ModelRunner):
    def load_input(self, batch_data):
        if self.is_ixrt_backend():
            err, = cuda.cuMemcpyHtoD(self.inputs[0]["allocation"], batch_data, batch_data.nbytes)
            assert(err == cuda.CUresult.CUDA_SUCCESS)

        elif self.is_ort_backend():
            self.inputs[0]["allocation"] = batch_data
        else:
            raise

    def fetch_output(self):
        if self.is_ixrt_backend():
            output = self.outputs[0]
            result = np.zeros(output["shape"],output["dtype"])
            err, = cuda.cuMemcpyDtoH(result, output["allocation"], output["nbytes"])
            assert(err == cuda.CUresult.CUDA_SUCCESS)
            return [result]

        elif self.is_ort_backend():
            return [output["allocation"] for output in self.outputs]
        else:
            raise

class ClassificationClipRunner(ClassificationRunner):
    def load_input(self, input_id, image, attention):
        if self.is_ixrt_backend():
            err, = cuda.cuMemcpyHtoD(self.inputs[0]["allocation"], input_id, input_id.nbytes)
            assert(err == cuda.CUresult.CUDA_SUCCESS)
            err, = cuda.cuMemcpyHtoD(self.inputs[1]["allocation"], image, image.nbytes)
            assert(err == cuda.CUresult.CUDA_SUCCESS)
            err, = cuda.cuMemcpyHtoD(self.inputs[2]["allocation"], attention, attention.nbytes)
            assert(err == cuda.CUresult.CUDA_SUCCESS)

        elif self.is_ort_backend():
            self.inputs[0]["allocation"] = input_id
            self.inputs[1]["allocation"] = image
            self.inputs[2]["allocation"] = attention
        else:
            raise

def main(config):
    if "MobileNet_v1" in config.engine_file:
        dataloader = getmobilenetv1dataloader(config.datasets_dir, config.loop_count, config.bsz, img_sz=config.imgsz)
    elif "/clip/" in config.engine_file:
        input_dict = {}
        input_name_list = []

        for input_info in ['input_ids:1000,22', 'pixel_values:32,3,224,224', 'attention_mask:1000,22']:
            input_name, input_shape = input_info.split(":")
            shape = tuple([int(s) for s in input_shape.split(",")])
            input_name_list.append(input_name)
            input_dict[input_name] = shape
        dataloader = getclipdataloader(config.bsz, config.datasets_dir, input_dict)
    else:
        dataloader = getdataloader(config.datasets_dir, config.loop_count, config.bsz, img_sz=config.imgsz)

    logger = tensorrt.Logger(tensorrt.Logger.ERROR)
    if "/clip/" in config.engine_file:
        runner = ClassificationClipRunner(config.engine_file, logger)
    else:
        runner = ClassificationRunner(config.engine_file, logger)

    # Inference
    if config.test_mode == "FPS":
        # Warm up
        if config.warm_up > 0:
            print("\nWarm Start.")
            for i in range(config.warm_up):
                runner.run()
            print("Warm Done.")
        torch.cuda.synchronize()
        start_time = time.time()

        for i in range(config.loop_count):
            runner.run()

        torch.cuda.synchronize()
        end_time = time.time()
        forward_time = end_time - start_time

        num_samples = 50000
        if config.loop_count * config.bsz < num_samples:
            num_samples = config.loop_count * config.bsz
        fps = num_samples / forward_time

        print("FPS : ", fps)
        print(f"Performance Check : Test {fps} >= target {config.fps_target}")
        if fps >= config.fps_target:
            print("pass!")
            exit()
        else:
            print("failed!")
            exit(1)

    elif config.test_mode == "ACC":
        total_sample = 0
        acc_top1, acc_top5 = 0, 0

        with tqdm(total= len(dataloader)) as _tqdm:
            if "/clip/" in config.engine_file:
                for idx, (input_id, image, attention, batch_label) in enumerate(dataloader):
                    image = image.astype(runner.inputs[1]["dtype"])
                    image = np.ascontiguousarray(image)
                    total_sample += image.shape[0]

                    runner.load_input(input_id, image, attention)
                    runner.run()
                    output = runner.fetch_output()[0]

                    batch_top1, batch_top5 = eval_batch(output, batch_label)
                    acc_top1 += batch_top1
                    acc_top5 += batch_top5

                    _tqdm.set_postfix(acc_1='{:.4f}'.format(acc_top1/total_sample),
                                      acc_5='{:.4f}'.format(acc_top5/total_sample))
                    _tqdm.update(1)
            else:
                for idx, (batch_data, batch_label) in enumerate(dataloader):
                    batch_data = batch_data.numpy().astype(runner.inputs[0]["dtype"])
                    batch_data = np.ascontiguousarray(batch_data)
                    total_sample += batch_data.shape[0]
                    
                    runner.load_input(batch_data)
                    runner.run()
                    output = runner.fetch_output()[0]

                    # squeeze output shape [32,1000,1,1] to [32,1000] for mobilenet_v2 model
                    if len(output.shape) == 4:
                        output = output.squeeze(axis=(2,3))

                    batch_top1, batch_top5 = eval_batch(output, batch_label)
                    acc_top1 += batch_top1
                    acc_top5 += batch_top5

                    _tqdm.set_postfix(acc_1='{:.4f}'.format(acc_top1/total_sample),
                                        acc_5='{:.4f}'.format(acc_top5/total_sample))
                    _tqdm.update(1)

        print(F"Acc@1 : {acc_top1/total_sample} = {acc_top1}/{total_sample}")
        print(F"Acc@5 : {acc_top5/total_sample} = {acc_top5}/{total_sample}")
        acc1 = acc_top1/total_sample
        print(f"Accuracy Check : Test {acc1} >= target {config.acc_target}")
        if acc1 >= config.acc_target:
            print("pass!")
            exit()
        else:
            print("failed!")
            exit(1)

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

    config = parser.parse_args()
    return config

if __name__ == "__main__":
    config = parse_config()
    main(config)
