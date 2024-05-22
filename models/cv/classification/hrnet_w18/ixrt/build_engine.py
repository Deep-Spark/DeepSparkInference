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
import cv2
import argparse
import numpy as np

import torch
import tensorrt
from calibration_dataset import getdataloader
import cuda.cudart as cudart

def assertSuccess(err):
    assert(err == cudart.cudaError_t.cudaSuccess)

class EngineCalibrator(tensorrt.IInt8EntropyCalibrator2):

    def __init__(self, cache_file, datasets_dir, loop_count=10, bsz=1, img_sz=224):
        super().__init__()
        self.cache_file = cache_file
        self.image_batcher  = getdataloader(datasets_dir, loop_count, batch_size=bsz, img_sz=img_sz)
        self.batch_generator = iter(self.image_batcher)
        size = img_sz*img_sz*3*bsz
        __import__('pdb').set_trace()
        err, self.batch_allocation = cudart.cudaMalloc(size)
        assertSuccess(err)

    def __del__(self):
        err,= cudart.cudaFree(self.batch_allocation)
        assertSuccess(err)

    def get_batch_size(self):
        return self.image_batcher.batch_size

    def get_batch(self, names):
        try:
            batch, _ = next(self.batch_generator)
            batch = batch.numpy()
            __import__('pdb').set_trace()
            cudart.cudaMemcpy(self.batch_allocation,
                              np.ascontiguousarray(batch),
                              batch.nbytes,
                              cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
            return [int(self.batch_allocation)]
        except StopIteration:
            return None

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

def main(config):
    IXRT_LOGGER = tensorrt.Logger(tensorrt.Logger.VERBOSE)
    builder = tensorrt.Builder(IXRT_LOGGER)
    EXPLICIT_BATCH = 1 << (int)(tensorrt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)
    build_config = builder.create_builder_config()
    parser = tensorrt.OnnxParser(network, IXRT_LOGGER)
    parser.parse_from_file(config.model)

    precision = tensorrt.BuilderFlag.INT8 if config.precision == "int8" else tensorrt.BuilderFlag.FP16
    print("precision : ", precision)
    build_config.set_flag(precision)
    if config.precision == "int8":
        build_config.int8_calibrator = EngineCalibrator("int8_cache", config.datasets_dir)

    plan = builder.build_serialized_network(network, build_config)
    engine_file_path = config.engine
    with open(engine_file_path, "wb") as f:
        f.write(plan)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--precision", type=str, choices=["float16", "int8", "float32"], default="int8",
            help="The precision of datatype")
    parser.add_argument("--engine", type=str, default=None)
    parser.add_argument(
        "--datasets_dir",
        type=str,
        default="",
        help="ImageNet dir",
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # cali = EngineCalibrator("tmp", "/home/qiang.zhang/data/imagenet_val/")
    # print(cali.get_batch_size())
    # print(cali.get_batch("hello"))
    args = parse_args()
    main(args)
