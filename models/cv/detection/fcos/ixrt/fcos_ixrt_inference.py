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
import sys
from tqdm import tqdm
import numpy as np

import argparse

import torch
import mmcv
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from mmdet.core import bbox2result
import cv2
import numpy as np
import onnxruntime as rt

import time

import os 
import copy
from common import create_engine_context, get_io_bindings
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt
from tensorrt import Dims

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

    # Prepare the output data
    outputs_651 = np.zeros(outputs[0]["shape"], outputs[0]["dtype"])
    outputs_766 = np.zeros(outputs[1]["shape"], outputs[1]["dtype"])
    outputs_881 = np.zeros(outputs[2]["shape"], outputs[2]["dtype"])
    outputs_996 = np.zeros(outputs[3]["shape"], outputs[3]["dtype"])
    outputs_1111 = np.zeros(outputs[4]["shape"], outputs[4]["dtype"])
    outputs_713 = np.zeros(outputs[5]["shape"], outputs[5]["dtype"])
    outputs_828 = np.zeros(outputs[6]["shape"], outputs[6]["dtype"])
    outputs_943 = np.zeros(outputs[7]["shape"], outputs[7]["dtype"])
    outputs_1058 = np.zeros(outputs[8]["shape"], outputs[8]["dtype"])
    outputs_1173 = np.zeros(outputs[9]["shape"], outputs[9]["dtype"])
    outputs_705 = np.zeros(outputs[10]["shape"], outputs[10]["dtype"])
    outputs_820 = np.zeros(outputs[11]["shape"], outputs[11]["dtype"])
    outputs_935 = np.zeros(outputs[12]["shape"], outputs[12]["dtype"])
    outputs_1050 = np.zeros(outputs[13]["shape"], outputs[13]["dtype"])
    outputs_1165 = np.zeros(outputs[14]["shape"], outputs[14]["dtype"])

    preds = []
    for batch in tqdm(dataloader):
        image = batch['img'][0].data.numpy()
        image = image.astype(inputs[0]["dtype"])
        # Set input
        image = np.ascontiguousarray(image) 
        cuda.memcpy_htod(inputs[0]["allocation"], image)
        context.execute_v2(allocations)
        # # Fetch output
        cuda.memcpy_dtoh(outputs_651, outputs[0]["allocation"])
        cuda.memcpy_dtoh(outputs_766, outputs[1]["allocation"])
        cuda.memcpy_dtoh(outputs_881, outputs[2]["allocation"])
        cuda.memcpy_dtoh(outputs_996, outputs[3]["allocation"])
        cuda.memcpy_dtoh(outputs_1111, outputs[4]["allocation"])
        cuda.memcpy_dtoh(outputs_713, outputs[5]["allocation"])
        cuda.memcpy_dtoh(outputs_828, outputs[6]["allocation"])
        cuda.memcpy_dtoh(outputs_943, outputs[7]["allocation"])
        cuda.memcpy_dtoh(outputs_1058, outputs[8]["allocation"])
        cuda.memcpy_dtoh(outputs_1173, outputs[9]["allocation"])
        cuda.memcpy_dtoh(outputs_705, outputs[10]["allocation"])
        cuda.memcpy_dtoh(outputs_820, outputs[11]["allocation"])
        cuda.memcpy_dtoh(outputs_935, outputs[12]["allocation"])
        cuda.memcpy_dtoh(outputs_1050, outputs[13]["allocation"])
        cuda.memcpy_dtoh(outputs_1165, outputs[14]["allocation"])

        cls_score = []
        box_reg = []
        score_factors = []
        cls_score.append(torch.from_numpy(outputs_651))
        cls_score.append(torch.from_numpy(outputs_766))
        cls_score.append(torch.from_numpy(outputs_881))
        cls_score.append(torch.from_numpy(outputs_996))
        cls_score.append(torch.from_numpy(outputs_1111))

        box_reg.append(torch.from_numpy(outputs_713))
        box_reg.append(torch.from_numpy(outputs_828))
        box_reg.append(torch.from_numpy(outputs_943))
        box_reg.append(torch.from_numpy(outputs_1058))
        box_reg.append(torch.from_numpy(outputs_1173))

        score_factors.append(torch.from_numpy(outputs_705))
        score_factors.append(torch.from_numpy(outputs_820))
        score_factors.append(torch.from_numpy(outputs_935))
        score_factors.append(torch.from_numpy(outputs_1050))
        score_factors.append(torch.from_numpy(outputs_1165))

        cls_score.sort(key=lambda x: x.shape[3],reverse=True)
        box_reg.sort(key=lambda x: x.shape[3],reverse=True)       
        score_factors.sort(key=lambda x: x.shape[3],reverse=True)

        pred = model.bbox_head.get_bboxes(cls_score, box_reg, score_factors=score_factors, img_metas=batch['img_metas'][0].data[0], rescale=True)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, model.bbox_head.num_classes)
            for det_bboxes, det_labels in pred
        ]
        preds.extend(bbox_results)
    eval_results = dataset.evaluate(preds, metric=['bbox'])
    print(eval_results)
    
    map50 = eval_results['bbox_mAP_50']
    return map50   

def parse_args():
    parser = argparse.ArgumentParser()
    # engine args
    parser.add_argument("--engine", type=str, default="./r50_fcos.engine")
    parser.add_argument("--cfg_file", type=str, default="fcos_r50_caffe_fpn_gn-head_1x_coco.py")
    parser.add_argument("--data_path", type=str, default="/home/datasets/cv/coco")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_file", type=str, default="/home/fangjian.hu/workspace/ixrt/data/fcos_test/test_800.jpg")
    parser.add_argument("--warp_up", type=int, default=40)
    parser.add_argument("--loop_count", type=int, default=50)
    
    parser.add_argument("--target_map", default=0.56, type=float, help="target map0.5")
    parser.add_argument("--target_fps", default=50, type=float, help="target fps")
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
        map50= eval_coco(args,inputs, outputs, allocations, context)
        
        print("="*40)
        print("MAP50:{0}".format(round(map50,3)))
        print("="*40)
        print(f"Check MAP50 Test : {round(map50,3)}  Target:{args.target_map} State : {'Pass' if round(map50,3) >= args.target_map else 'Fail'}")
        status_map = check_target(map50, args.target_map)
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