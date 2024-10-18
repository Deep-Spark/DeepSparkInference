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

import os
import sys
import cv2
import time
import json
import math
import argparse
import numpy as np
from copy import deepcopy

import torch
import tensorrt
from tensorrt import Dims
from cuda import cuda, cudart

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from common import CocoValDataset
from modules.keypoints import extract_keypoints, group_keypoints


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', type=str, default="annotations/person_keypoints_val2017.json", help='path to json with keypoints val labels')
    parser.add_argument('--output-name', type=str, default='detections.json', help='name of output json file with detected keypoints')
    parser.add_argument('--images-folder', type=str, default="val2017/", help='path to COCO val images folder')
    parser.add_argument('--multiscale', action='store_true', help='average inference results over multiple scales')
    parser.add_argument("--data_type", type=str, default="float16", help="int8 float16")
    parser.add_argument("--model_type", type=str, default="lightweight_openose", help="EfficientNet ResNet50 Vgg16 MobileNet")
    parser.add_argument("--test_mode", type=str, default="MAP", help="FPS MAP")
    parser.add_argument("--graph_file", type=str, help="graph file path")
    parser.add_argument("--weights_file",type=str, help="weights file path")
    parser.add_argument("--engine_file", type=str, help="engine file path")
    parser.add_argument("--quant_file", type=str, help="weights file path")
    parser.add_argument("--datasets_dir", type=str, default="", help="coco pose dir")
    parser.add_argument("--warm_up", type=int, default=-1, help="warm_up times")
    parser.add_argument("--bsz", type=int, default=1, help="test batch size")
    parser.add_argument("--imgh", type=int, default=256, help="inference size h")
    parser.add_argument("--max_imgw", type=int, default=456, help="inference size max w")
    parser.add_argument("--use_async", action="store_true")
    parser.add_argument("--fixed_shape", action="store_true")
    parser.add_argument("--device", type=int, default=0, help="cuda device, i.e. 0 or 0,1,2,3,4")
    parser.add_argument("--map_target", type=float, default=-1.0)
    config = parser.parse_args()
    return config


def openpose_trtapi_ixrt(config):
    engine_file = config.engine_file
    datatype = tensorrt.DataType.FLOAT
    host_mem = tensorrt.IHostMemory
    logger = tensorrt.Logger(tensorrt.Logger.ERROR)
    with open(engine_file, "rb") as f, tensorrt.Runtime(logger) as runtime:
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


def run_coco_eval(gt_file_path, dt_file_path):
    annotation_type = 'keypoints'
    print('Running test for {} results.'.format(annotation_type))

    coco_gt = COCO(gt_file_path)
    coco_dt = coco_gt.loadRes(dt_file_path)

    result = COCOeval(coco_gt, coco_dt, annotation_type)
    result.evaluate()
    result.accumulate()
    result.summarize()


def normalize(img, img_mean, img_scale):
    img = np.array(img, dtype=np.float32)
    img = (img - img_mean) * img_scale
    return img


def pad_width(img, stride, pad_value, min_dims):
    h, w, _ = img.shape
    h = min(min_dims[0], h)
    min_dims[0] = math.ceil(min_dims[0] / float(stride)) * stride
    min_dims[1] = max(min_dims[1], w)
    min_dims[1] = math.ceil(min_dims[1] / float(stride)) * stride
    pad = []
    pad.append(int(math.floor((min_dims[0] - h) / 2.0)))
    pad.append(int(math.floor((min_dims[1] - w) / 2.0)))
    pad.append(int(min_dims[0] - h - pad[0]))
    pad.append(int(min_dims[1] - w - pad[1]))
    padded_img = cv2.copyMakeBorder(img, pad[0], pad[2], pad[1], pad[3],
                                    cv2.BORDER_CONSTANT, value=pad_value)
    return padded_img, pad


def convert_to_coco_format(pose_entries, all_keypoints):
    coco_keypoints = []
    scores = []
    for n in range(len(pose_entries)):
        if len(pose_entries[n]) == 0:
            continue
        keypoints = [0] * 17 * 3
        to_coco_map = [0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
        person_score = pose_entries[n][-2]
        position_id = -1
        for keypoint_id in pose_entries[n][:-2]:
            position_id += 1
            if position_id == 1:  # no 'neck' in COCO
                continue

            cx, cy, score, visibility = 0, 0, 0, 0  # keypoint not found
            if keypoint_id != -1:
                cx, cy, score = all_keypoints[int(keypoint_id), 0:3]
                cx = cx + 0.5
                cy = cy + 0.5
                visibility = 1
            keypoints[to_coco_map[position_id] * 3 + 0] = cx
            keypoints[to_coco_map[position_id] * 3 + 1] = cy
            keypoints[to_coco_map[position_id] * 3 + 2] = visibility
        coco_keypoints.append(keypoints)
        scores.append(person_score * max(0, (pose_entries[n][-1] - 1)))  # -1 for 'neck'
    return coco_keypoints, scores


def infer(img, scales, engine, context, base_height, stride, config, pad_value=(0, 0, 0), img_mean=(128, 128, 128), img_scale=1/256):

    input_name = "data"
    output_names = [
        "stage_0_output_1_heatmaps",
        "stage_0_output_0_pafs",
        "stage_1_output_1_heatmaps",
        "stage_1_output_0_pafs"
    ]

    normed_img = normalize(img, img_mean, img_scale)
    height, width, _ = normed_img.shape
    scales_ratios = [scale * base_height / float(height) for scale in scales]
    avg_heatmaps = np.zeros((height, width, 19), dtype=np.float32)
    avg_pafs = np.zeros((height, width, 38), dtype=np.float32)

    for ratio in scales_ratios:
        scaled_img = cv2.resize(normed_img, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
        min_dims = [base_height, max(scaled_img.shape[1], base_height)]
        padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)
        hh, ww ,_ = padded_img.shape

        data_batch = padded_img.transpose(2, 0, 1)
        data_batch = np.ascontiguousarray(data_batch.reshape(1, *data_batch.shape).astype(np.float32))

        input_shape = [1, 3, hh, ww]
        input_idx = engine.get_binding_index(input_name)
        context.set_binding_shape(input_idx, Dims(input_shape))

        inputs, outputs, allocations = setup_io_bindings(engine, context)

        pred_outputs = []
        for output in outputs:
            pred_outputs.append(np.zeros(output["shape"], output["dtype"]))
        err, = cuda.cuMemcpyHtoD(inputs[0]["allocation"], data_batch, data_batch.nbytes)
        assert(err == cuda.CUresult.CUDA_SUCCESS)

        if config.use_async:
            stream = cuda.Stream()
            context.execute_async_v2(allocations, stream.handle)
            stream.synchronize()
        else:
            context.execute_v2(allocations)

        for i, pred_output in enumerate(pred_outputs):
            err, = cuda.cuMemcpyDtoH(pred_output, outputs[i]["allocation"], outputs[i]["nbytes"])
            assert(err == cuda.CUresult.CUDA_SUCCESS)

        heatmaps = deepcopy(pred_outputs[2][0].transpose(1, 2, 0)).astype(np.float32)
        heatmaps = cv2.resize(heatmaps[:,:,:19], (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
        heatmaps = heatmaps[pad[0]:heatmaps.shape[0] - pad[2], pad[1]:heatmaps.shape[1] - pad[3]:, :]
        heatmaps = cv2.resize(heatmaps, (width, height), interpolation=cv2.INTER_CUBIC)
        avg_heatmaps = avg_heatmaps + heatmaps / len(scales_ratios)

        pafs = deepcopy(pred_outputs[3][0].transpose(1, 2, 0)).astype(np.float32)
        pafs = cv2.resize(pafs[:,:,:38], (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
        pafs = pafs[pad[0]:pafs.shape[0] - pad[2], pad[1]:pafs.shape[1] - pad[3], :]
        pafs = cv2.resize(pafs, (width, height), interpolation=cv2.INTER_CUBIC)
        avg_pafs = avg_pafs + pafs / len(scales_ratios)

    return avg_heatmaps, avg_pafs


def evaluate(labels, output_name, images_folder, engine, context, config, multiscale=False, visualize=False):
    base_height = 368
    scales = [1]
    if multiscale:
        scales = [0.5, 1.0, 1.5, 2.0]
    stride = 8

    dataset = CocoValDataset(labels, images_folder)
    coco_result = []
    for i, sample in enumerate(dataset):
        file_name = sample['file_name']
        img = sample['img']
        if i % 20 == 1:
            print("{}/{} img shape {}".format(i, len(dataset), img.shape))

        avg_heatmaps, avg_pafs = infer(img, scales, engine, context, base_height, stride, config)
        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(18):  # 19th for bg
            total_keypoints_num += extract_keypoints(avg_heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)
        # print("total_keypoints_num ",total_keypoints_num)
        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, avg_pafs)

        coco_keypoints, scores = convert_to_coco_format(pose_entries, all_keypoints)
        # print(coco_keypoints)

        image_id = int(file_name[0:file_name.rfind('.')])
        for idx in range(len(coco_keypoints)):
            coco_result.append({
                'image_id': image_id,
                'category_id': 1,  # person
                'keypoints': coco_keypoints[idx],
                'score': scores[idx]
            })
            

        # if i<100 and total_keypoints_num > 0:
        #     for keypoints in coco_keypoints:
        #         for idx in range(len(keypoints) // 3):
        #             cv2.circle(img, (int(keypoints[idx * 3]), int(keypoints[idx * 3 + 1])),
        #                        3, (255, 0, 255), -1)
        #     save_name = "{}.jpg".format(i)
        #     cv2.imwrite(save_name, img)


    with open(output_name, 'w') as f:
        json.dump(coco_result, f, indent=4)

    run_coco_eval(labels, output_name)


def main(config):
    engine, context = openpose_trtapi_ixrt(config)
    print(" config and load model ok...")
    evaluate(config.labels, config.output_name, config.images_folder, engine, context, config)
    print(" done ...")
    

if __name__ == '__main__':
    config = parse_config()
    main(config)
    

