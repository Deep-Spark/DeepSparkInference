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

import argparse
import json
import os
import re
import time
from tqdm import tqdm

import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import torch
import tensorrt

from calibration_dataset import getdataloader
from common import eval_batch, create_engine_context, get_io_bindings


def main(config):
    dataloader = getdataloader(config.datasets_dir, config.loop_count, config.bsz, img_sz=config.imgsz)

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

        ## Prepare the output data
        output = np.zeros(outputs[0]["shape"], outputs[0]["dtype"])
        print(f"output shape : {output.shape} output type : {output.dtype}")

        total_sample = 0
        acc_top1, acc_top5 = 0, 0

        start_time = time.time()
        with tqdm(total= len(dataloader)) as _tqdm:
            for idx, (batch_data, batch_label) in enumerate(dataloader):
                batch_data = batch_data.numpy().astype(inputs[0]["dtype"])
                batch_data = np.ascontiguousarray(batch_data)
                total_sample += batch_data.shape[0]

                cuda.memcpy_htod(inputs[0]["allocation"], batch_data)
                context.execute_v2(allocations)
                cuda.memcpy_dtoh(output, outputs[0]["allocation"])

                # squeeze output shape [32,1000,1,1] to [32,1000] for mobilenet_v2 model
                if len(output.shape) == 4:
                    output = output.squeeze(axis=(2,3))

                batch_top1, batch_top5 = eval_batch(output, batch_label)
                acc_top1 += batch_top1
                acc_top5 += batch_top5

                _tqdm.set_postfix(acc_1='{:.4f}'.format(acc_top1/total_sample),
                                    acc_5='{:.4f}'.format(acc_top5/total_sample))
                _tqdm.update(1)
        end_time = time.time()
        end2end_time = end_time - start_time

        print(F"E2E time : {end2end_time:.3f} seconds")
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
    parser.add_argument("--test_mode", type=str, default="FPS", help="FPS ACC")
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
