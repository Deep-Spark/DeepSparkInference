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
import tensorrt
import torch
import torchvision
import numpy as np
from tensorrt import Dims
from cuda import cuda, cudart
from tqdm import tqdm
from mmdet.registry import RUNNERS
from mmengine.config import Config

from common import create_engine_context, get_io_bindings

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

    parser.add_argument("--input_name", 
                        type=str, 
                        required=True, 
                        help="input name of the model.")
    
    parser.add_argument("--warmup", 
                        type=int, 
                        default=3, 
                        help="number of warmup before test.")           
    
    parser.add_argument("--acc_target",
                        type=float,
                        default=None,
                        help="Model inference Accuracy target.")
    
    parser.add_argument("--fps_target",
                        type=float,
                        default=700.0,
                        help="Model inference FPS target.")

    parser.add_argument("--perf_only",
                        type=bool,
                        default=False,
                        help="Run performance test only")
    
    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    batch_size = args.batchsize

    host_mem = tensorrt.IHostMemory
    logger = tensorrt.Logger(tensorrt.Logger.ERROR)

    # Load Engine && I/O bindings
    engine, context = create_engine_context(args.engine, logger)
    inputs, outputs, allocations = get_io_bindings(engine)
    
    if args.warmup > 0:
        print("\nWarm Start.")
        for i in range(args.warmup):
            context.execute_v2(allocations)
        print("Warm Done.")

    # just run perf test
    if args.perf_only:
        torch.cuda.synchronize()
        start_time = time.time()

        for i in range(1000):
            context.execute_v2(allocations)

        torch.cuda.synchronize()
        end_time = time.time()
        forward_time = end_time - start_time
        num_samples = 1000 * args.batchsize
        fps = num_samples / forward_time

        print("FPS : ", fps)
        print(f"Performance Check : Test {fps} >= target {args.fps_target}")
        if fps >= args.fps_target:
            print("pass!")
            exit()
        else:
            print("failed!")
            exit(1)
    else:
        # Runner config
        cfg = Config.fromfile("centernet_r18_8xb16-crop512-140e_coco.py")
        cfg.work_dir = "./"

        cfg['test_dataloader']['batch_size'] = batch_size
        cfg['test_dataloader']['dataset']['data_root'] = args.datasets
        cfg['test_dataloader']['dataset']['data_prefix']['img'] = 'images/val2017/'
        cfg['test_evaluator']['ann_file'] = os.path.join(args.datasets, 'annotations/instances_val2017.json')

        runner = RUNNERS.build(cfg)

        start_time = time.time()
        for input_data in tqdm(runner.test_dataloader):
            
            input_data = runner.model.data_preprocessor(input_data, False)
            image = input_data['inputs'].cpu()
            image = image.numpy().astype(inputs[0]["dtype"])

            pad_batch = len(image) != batch_size
            if pad_batch:
                origin_size = len(image)
                image = np.resize(image, (batch_size, *image.shape[1:]))
            
            image = np.ascontiguousarray(image)

            (err,) = cudart.cudaMemcpy(
                inputs[0]["allocation"],
                image,
                image.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            )
            assert err == cudart.cudaError_t.cudaSuccess
            # cuda.memcpy_htod(inputs[0]["allocation"], batch_data)
            context.execute_v2(allocations)
            
            results = []
            
            for i in range(len(outputs)):
                output = np.zeros(outputs[i]["shape"], outputs[i]["dtype"])
                (err,) = cudart.cudaMemcpy(
                    output,
                    outputs[i]["allocation"],
                    outputs[i]["nbytes"],
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                )
                assert err == cudart.cudaError_t.cudaSuccess

                if pad_batch:
                    output = output[:origin_size]

                output = torch.from_numpy(output)
                results.append(output)

            batch_img_metas = [
                data_samples.metainfo for data_samples in input_data['data_samples']
            ]
            
            results_list = runner.model.bbox_head.predict_by_feat([results[0]], [results[1]], [results[2]], batch_img_metas=batch_img_metas, rescale=True)

            batch_data_samples = runner.model.add_pred_to_datasample(input_data['data_samples'], results_list)

            runner.test_evaluator.process(data_samples=batch_data_samples, data_batch=input_data)

        end_time = time.time()
        e2e_time = end_time - start_time
        print(F"E2E time : {e2e_time:.3f} seconds")
        metrics = runner.test_evaluator.evaluate(len(runner.test_dataloader.dataset))

if __name__ == "__main__":
    main()