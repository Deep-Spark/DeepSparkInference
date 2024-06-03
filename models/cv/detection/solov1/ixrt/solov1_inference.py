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
import os
import time
from typing import Tuple
import cv2
import numpy as np
import torch
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
import mmcv
from mmdet.core import encode_mask_results
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt
from tqdm import tqdm
import numpy as np
import sys
from common import create_engine_context, get_io_bindings

def check_target(inference, target):
    satisfied = False
    if inference > target:
        satisfied = True  
    return satisfied 
    

def get_dataloder(args):
    cfg_path = args.cfg_file
    cfg = mmcv.Config.fromfile(cfg_path)
    datasets_path = args.data_path
    cfg['data']['val']['img_prefix'] = os.path.join(datasets_path, 'val2017')
    cfg['data']['val']['ann_file'] = os.path.join(datasets_path, 'annotations/instances_val2017.json')   
    dataset = build_dataset(cfg.data.val)
    data_loader = build_dataloader(dataset, samples_per_gpu=args.batch_size, workers_per_gpu=args.num_workers, shuffle=False)
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    return dataset, data_loader, model


def eval_coco(args, inputs, outputs, allocations, context):
    dataset, dataloader, model = get_dataloder(args)
    outputs_0 = np.zeros(outputs[0]["shape"], outputs[0]["dtype"])
    outputs_1 = np.zeros(outputs[1]["shape"], outputs[1]["dtype"])
    outputs_2 = np.zeros(outputs[2]["shape"], outputs[2]["dtype"])
    outputs_3 = np.zeros(outputs[3]["shape"], outputs[3]["dtype"])
    outputs_4 = np.zeros(outputs[4]["shape"], outputs[4]["dtype"])
    outputs_5 = np.zeros(outputs[5]["shape"], outputs[5]["dtype"])
    outputs_6 = np.zeros(outputs[6]["shape"], outputs[6]["dtype"])
    outputs_7 = np.zeros(outputs[7]["shape"], outputs[7]["dtype"])
    outputs_8 = np.zeros(outputs[8]["shape"], outputs[8]["dtype"])
    outputs_9 = np.zeros(outputs[9]["shape"], outputs[9]["dtype"])

    results = []
    for batch in tqdm(dataloader):
        image = batch['img'][0].data.numpy()
        img_metas = batch['img_metas'][0].data[0]
        # Set input
        image = np.ascontiguousarray(image) 
        cuda.memcpy_htod(inputs[0]["allocation"], image)
        context.execute_v2(allocations)
        # # Fetch output
        cuda.memcpy_dtoh(outputs_0, outputs[0]["allocation"])
        cuda.memcpy_dtoh(outputs_1, outputs[1]["allocation"])
        cuda.memcpy_dtoh(outputs_2, outputs[2]["allocation"])
        cuda.memcpy_dtoh(outputs_3, outputs[3]["allocation"])
        cuda.memcpy_dtoh(outputs_4, outputs[4]["allocation"])
        cuda.memcpy_dtoh(outputs_5, outputs[5]["allocation"])
        cuda.memcpy_dtoh(outputs_6, outputs[6]["allocation"])
        cuda.memcpy_dtoh(outputs_7, outputs[7]["allocation"])
        cuda.memcpy_dtoh(outputs_8, outputs[8]["allocation"])
        cuda.memcpy_dtoh(outputs_9, outputs[9]["allocation"])

        mask_preds = []
        cls_preds = []

        mask_preds.append(torch.from_numpy(outputs_0))
        mask_preds.append(torch.from_numpy(outputs_1))
        mask_preds.append(torch.from_numpy(outputs_2))
        mask_preds.append(torch.from_numpy(outputs_3))
        mask_preds.append(torch.from_numpy(outputs_4))
        cls_preds.append(torch.from_numpy(outputs_5))
        cls_preds.append(torch.from_numpy(outputs_6))
        cls_preds.append(torch.from_numpy(outputs_7))
        cls_preds.append(torch.from_numpy(outputs_8))
        cls_preds.append(torch.from_numpy(outputs_9))
        mask_preds.sort(key=lambda x: x.shape[1], reverse=True)
        cls_preds.sort(key=lambda x: x.shape[2], reverse=True)
        results_list = model.mask_head.get_results(mask_preds, cls_preds, img_metas)
        format_results_list = []
        for result in results_list:
            format_results_list.append(model.format_results(result))
            
        if isinstance(format_results_list[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in format_results_list]
        results.extend(result)        
    eval_results = dataset.evaluate(results, metric=['segm'])
    print(eval_results)
    segm_mAP = eval_results['segm_mAP']
    return segm_mAP
    

def parse_args():
    parser = argparse.ArgumentParser()
    # engine args
    parser.add_argument("--engine", type=str, default="")
    parser.add_argument("--cfg_file", type=str, default="")
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--warp_up", type=int, default=40)
    parser.add_argument("--loop_count", type=int, default=50)
    parser.add_argument("--target_map", default=0.331, type=float, help="target map")
    parser.add_argument("--target_fps", default=15, type=float, help="target fps")
    parser.add_argument("--task", default="precision", type=str, help="precision or pref")
    args = parser.parse_args()
    return args


def main():
    args= parse_args()
    host_mem = tensorrt.IHostMemory
    logger = tensorrt.Logger(tensorrt.Logger.ERROR)

    # Load Engine
    engine, context = create_engine_context(args.engine, logger)
    inputs, outputs, allocations = get_io_bindings(engine)
    
    if args.task=="precision":
        segm_mAP= eval_coco(args,inputs, outputs, allocations, context)
        
        print("="*40)
        print("segm_mAP:{0}".format(round(segm_mAP,3)))
        print("="*40)
        print(f"Check segm_mAP Test : {round(segm_mAP,3)}  Target:{args.target_map} State : {'Pass' if round(segm_mAP,3) >= args.target_map else 'Fail'}")
        status_map = check_target(segm_mAP, args.target_map)
        sys.exit(int(not (status_map)))   
    else:
        torch.cuda.synchronize()
        start_time = time.time()
        for i in range(args.loop_count):
            context.execute_v2(allocations)  
        torch.cuda.synchronize()
        end_time = time.time()
        forward_time = end_time - start_time
        fps = args.loop_count * args.batch_size / forward_time
        print("="*40)
        print("fps:{0}".format(round(fps,2)))
        print("="*40)
        print(f"Check fps Test : {round(fps,3)}  Target:{args.target_fps} State : {'Pass' if  fps >= args.target_fps else 'Fail'}")
        status_fps = check_target(fps, args.target_fps)
        sys.exit(int(not (status_fps)))
    
if __name__ == "__main__":
    
    main()
