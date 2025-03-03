#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2025, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
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
from cuda import cuda, cudart
import torch
import onnx
import onnxruntime
import torchvision.datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import tensorrt

from wider_face_dataset import WiderFaceDetection, detection_collate
from models import *
from evaluation import evaluation
from common import create_engine_context, get_io_bindings

def post_process(args, prior_data, locs, confs, landms, resizes, img_files, net_h, net_w):
    import numpy as np
    from utils.box_utils import decode, decode_landm
    from utils.nms.py_cpu_nms import py_cpu_nms

    variances = 0.1, 0.2
    nms_threshold = 0.4
    confidence_threshold = 0.02

    scale = torch.Tensor([net_w, net_h, net_w, net_h])
    scale1 = torch.Tensor([net_w, net_h, net_w, net_h,
                            net_w, net_h, net_w, net_h,
                            net_w, net_h])

    for i, (loc, conf, landm, resize, img_name) in enumerate(zip(locs, confs, landms, resizes, img_files)):
        boxes = decode(loc.squeeze(0).data, prior_data, variances)
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()

        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landm = decode_landm(landm.squeeze(0).data, prior_data, variances)

        landm = landm * scale1 / resize
        landm = landm.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > confidence_threshold)[0]
        boxes = boxes[inds]
        landm = landm[inds]
        scores = scores[inds]

        order = scores.argsort()[::-1]
        boxes = boxes[order]
        landm = landm[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, nms_threshold)
        dets = dets[keep, :]
        landm = landm[keep]

        dets = np.concatenate((dets, landm), axis=1)

        # --------------------------------------------------------------------
        save_name = os.path.join(args.save_folder, f"{img_name[:-4]}.txt")
        dirname = os.path.dirname(save_name)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        with open(save_name, "w") as fd:
            bboxs = dets
            file_name = os.path.basename(save_name)[:-4] + "\n"
            bboxs_num = str(len(bboxs)) + "\n"
            fd.write(file_name)
            fd.write(bboxs_num)
            for box in bboxs:
                x = int(box[0])
                y = int(box[1])
                w = int(box[2]) - int(box[0])
                h = int(box[3]) - int(box[1])
                confidence = str(box[4])
                line = str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence + " \n"
                fd.write(line)


def main(config):
    dataset = WiderFaceDetection(config.datasets_dir)

    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=config.bsz,
        drop_last=True,
        num_workers=4,
        collate_fn=detection_collate
    )

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

        fps = config.loop_count * config.bsz / forward_time

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
        bbox_out0 = np.zeros(outputs[0]["shape"], outputs[0]["dtype"])
      
        bbox_out1 = np.zeros(outputs[3]["shape"], outputs[3]["dtype"])
     
        bbox_out2 = np.zeros(outputs[6]["shape"], outputs[6]["dtype"])
       
        cls_out0 = np.zeros(outputs[1]["shape"], outputs[1]["dtype"])
        
        cls_out1 = np.zeros(outputs[4]["shape"], outputs[4]["dtype"])
       
        cls_out2 = np.zeros(outputs[7]["shape"], outputs[7]["dtype"])
        
        ldm_out0 = np.zeros(outputs[2]["shape"], outputs[2]["dtype"])
        
        ldm_out1 = np.zeros(outputs[5]["shape"], outputs[5]["dtype"])
       
        ldm_out2 = np.zeros(outputs[8]["shape"], outputs[8]["dtype"])
        

        priorbox = PriorBox(image_size=(320, 320))
        priors = priorbox.forward()
        prior_data = priors.data

        for i, data in enumerate(tqdm(dataloader)):

            #print(i)
            imgs, rs, img_files = data
            batch_data = imgs.numpy().astype(inputs[0]["dtype"])
            batch_data = np.ascontiguousarray(batch_data)
            
            err, = cuda.cuMemcpyHtoD(inputs[0]["allocation"], batch_data, batch_data.nbytes)
            assert(err == cuda.CUresult.CUDA_SUCCESS)
            context.execute_v2(allocations)

            err, = cuda.cuMemcpyDtoH(bbox_out0, outputs[0]["allocation"], outputs[0]["nbytes"])
            assert(err == cuda.CUresult.CUDA_SUCCESS)
            bbox_out0_t = torch.from_numpy(bbox_out0.transpose([0,2,3,1])).reshape(bbox_out0.shape[0], -1, 4)

            err, = cuda.cuMemcpyDtoH(bbox_out1, outputs[3]["allocation"], outputs[3]["nbytes"])
            assert(err == cuda.CUresult.CUDA_SUCCESS)
            bbox_out1_t = torch.from_numpy(bbox_out1.transpose([0,2,3,1])).reshape(bbox_out1.shape[0], -1, 4)

            err, = cuda.cuMemcpyDtoH(bbox_out2, outputs[6]["allocation"], outputs[6]["nbytes"])
            assert(err == cuda.CUresult.CUDA_SUCCESS)
            bbox_out2_t = torch.from_numpy(bbox_out2.transpose([0,2,3,1])).reshape(bbox_out2.shape[0], -1, 4)

            err, = cuda.cuMemcpyDtoH(cls_out0, outputs[1]["allocation"], outputs[1]["nbytes"])
            assert(err == cuda.CUresult.CUDA_SUCCESS)
            cls_out0_t = torch.from_numpy(cls_out0.transpose([0,2,3,1])).reshape(cls_out0.shape[0], -1, 2)

            err, = cuda.cuMemcpyDtoH(cls_out1, outputs[4]["allocation"], outputs[4]["nbytes"])
            assert(err == cuda.CUresult.CUDA_SUCCESS)
            cls_out1_t = torch.from_numpy(cls_out1.transpose([0,2,3,1])).reshape(cls_out1.shape[0], -1, 2)

            err, = cuda.cuMemcpyDtoH(cls_out2, outputs[7]["allocation"], outputs[7]["nbytes"])
            assert(err == cuda.CUresult.CUDA_SUCCESS)
            cls_out2_t = torch.from_numpy(cls_out2.transpose([0,2,3,1])).reshape(cls_out2.shape[0], -1, 2)

            err, = cuda.cuMemcpyDtoH(ldm_out0, outputs[2]["allocation"], outputs[2]["nbytes"])
            assert(err == cuda.CUresult.CUDA_SUCCESS)
            ldm_out0_t = torch.from_numpy(ldm_out0.transpose([0,2,3,1])).reshape(ldm_out0.shape[0], -1, 10)
            
            err, = cuda.cuMemcpyDtoH(ldm_out1, outputs[5]["allocation"], outputs[5]["nbytes"])
            assert(err == cuda.CUresult.CUDA_SUCCESS)
            ldm_out1_t = torch.from_numpy(ldm_out1.transpose([0,2,3,1])).reshape(ldm_out1.shape[0], -1, 10)

            err, = cuda.cuMemcpyDtoH(ldm_out2, outputs[8]["allocation"], outputs[8]["nbytes"])
            assert(err == cuda.CUresult.CUDA_SUCCESS)
            ldm_out2_t = torch.from_numpy(ldm_out2.transpose([0,2,3,1])).reshape(ldm_out2.shape[0], -1, 10)

            

            bbox_regressions = torch.cat([bbox_out0_t, bbox_out1_t, bbox_out2_t], dim=1)
            classifications = torch.cat([cls_out0_t, cls_out1_t, cls_out2_t], dim=1)
            ldm_regressions = torch.cat([ldm_out0_t, ldm_out1_t, ldm_out2_t], dim=1)
            classifications = F.softmax(classifications, dim=-1)

            net_h, net_w = batch_data.shape[2], batch_data.shape[3]

            post_process(config, prior_data, bbox_regressions, classifications, ldm_regressions, rs, img_files, net_h, net_w)
        
        easy_AP = evaluation(config.pred_path, config.gt)

        if easy_AP >= config.acc_target:
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
    parser.add_argument("--save_folder", type=str, default=None)
    parser.add_argument("--pred_path", type=str, default=None)
    parser.add_argument('-g', '--gt', default=None)
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
