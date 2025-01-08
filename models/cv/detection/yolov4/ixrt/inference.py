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
import glob
import json
import os
import time
import sys
from tqdm import tqdm

import torch
import numpy as np
import tensorrt
from tensorrt import Dims
import pycuda.autoinit
import pycuda.driver as cuda

from coco_labels import coco80_to_coco91_class
from common import save2json, box_class85to6
from common import load_images, prepare_batch
from common import create_engine_context, setup_io_bindings
from common import scale_boxes, post_processing

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from load_ixrt_plugin import load_ixrt_plugin
load_ixrt_plugin()



def main(config):

    # Step1: Load dataloader
    images_path = load_images(config.eval_dir)
    dataloader = prepare_batch(images_path, config.bsz)

    # Step2: Load Engine
    input_name = "input"
    host_mem = tensorrt.IHostMemory
    logger = tensorrt.Logger(tensorrt.Logger.ERROR)
    engine, context = create_engine_context(config.model_engine, logger)
    input_idx = engine.get_binding_index(input_name)
    context.set_binding_shape(input_idx, Dims((config.bsz,3,config.imgsz,config.imgsz)))
    inputs, outputs, allocations = setup_io_bindings(engine, context)

    # Warm up
    if config.warm_up > 0:
        print("\nWarm Start.")
        for i in range(config.warm_up):
            context.execute_v2(allocations)
        print("Warm Done.")

    json_result = []
    forward_time = 0.0
    class_map = coco80_to_coco91_class()
    num_samples = 0
    start_time = time.time()
    # Step3: Run on coco dataset
    for batch_names, batch_images, batch_shapes in tqdm(zip(*dataloader)):
        batch_data = np.ascontiguousarray(batch_images)
        data_shape = batch_data.shape
        h, w = zip(*batch_shapes)
        batch_img_shape = [h, w]
        batch_img_id = [int(x.split('.')[0]) for x in batch_names]

        cur_bsz_sample = batch_images.shape[0]
        num_samples += cur_bsz_sample
        # Set input
        input_idx = engine.get_binding_index(input_name)
        context.set_binding_shape(input_idx, Dims(data_shape))
        inputs, outputs, allocations = setup_io_bindings(engine, context)

        cuda.memcpy_htod(inputs[0]["allocation"], batch_data)
        # Prepare the output data
        output = np.zeros(outputs[0]["shape"], outputs[0]["dtype"])
        # print(f"output shape : {output.shape} output type : {output.dtype}")

        # Forward
        start_time = time.time()
        context.execute_v2(allocations)
        end_time = time.time()
        forward_time += end_time - start_time

        if config.test_mode == "MAP":
            # Fetch output
            cuda.memcpy_dtoh(output, outputs[0]["allocation"])
            pred_boxes = post_processing(None, 0.001, 0.6, output)

            pred_results = []
            # Calculate pred box on raw shape
            for (pred_box, raw_shape) in zip(pred_boxes, batch_shapes):
                h, w = raw_shape
                if len(pred_box) == 0:continue    # no detection results
                pred_box = np.array(pred_box, dtype=np.float32)
                pred_box = scale_boxes((config.imgsz, config.imgsz), pred_box, raw_shape, use_letterbox=False)

                pred_results.append(pred_box.tolist())

            save2json(batch_img_id, pred_results, json_result, class_map)
    end_time = time.time()
    e2e_time = end_time - start_time
    fps = num_samples / forward_time

    if config.test_mode == "FPS":
        print("FPS : ", fps)
        print(f"Performance Check : Test {fps} >= target {config.fps_target}")
        if fps >= config.fps_target:
            print("pass!")
            exit()
        else:
            print("failed!")
            exit(1)

    if config.test_mode == "MAP":
        if len(json_result) == 0:
            print("Predict zero box!")
            exit(1)

        if not os.path.exists(config.pred_dir):
            os.makedirs(config.pred_dir)

        pred_json = os.path.join(
            config.pred_dir, f"{config.model_name}_{config.precision}_preds.json"
        )
        with open(pred_json, "w") as f:
            json.dump(json_result, f)

        anno_json = config.coco_gt
        anno = COCO(anno_json)  # init annotations api
        pred = anno.loadRes(pred_json)  # init predictions api
        eval = COCOeval(anno, pred, "bbox")

        eval.evaluate()
        eval.accumulate()
        print(
            f"==============================eval {config.model_name} {config.precision} coco map =============================="
        )
        eval.summarize()

        map, map50 = eval.stats[:2]
        print("MAP@0.5 : ", map50)
        print(f"Accuracy Check : Test {map50} >= target {config.map_target}")
        print(F"E2E time : {e2e_time:.3f} seconds")
        if map50 >= config.map_target:
            print("pass!")
            exit()
        else:
            print("failed!")
            exit(1)


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default="YOLOV4", help="YOLOV3 YOLOV4 YOLOV5 YOLOV7 YOLOX"
    )
    parser.add_argument("--precision", type=str, choices=["float16", "int8", "float32"], default="int8",
            help="The precision of datatype")
    parser.add_argument("--test_mode", type=str, default="FPS", help="FPS MAP")
    parser.add_argument(
        "--model_engine",
        type=str,
        default="",
        help="model engine path",
    )
    parser.add_argument(
        "--coco_gt",
        type=str,
        default="data/datasets/cv/coco2017/annotations/instances_val2017.json",
        help="coco instances_val2017.json",
    )
    parser.add_argument("--warm_up", type=int, default=3, help="warm_up count")
    parser.add_argument("--loop_count", type=int, default=-1, help="loop count")
    parser.add_argument(
        "--eval_dir",
        type=str,
        default="data/datasets/cv/coco2017/val2017",
        help="coco image dir",
    )
    parser.add_argument("--bsz", type=int, default=32, help="test batch size")
    parser.add_argument(
        "--imgsz",
        "--img",
        "--img-size",
        type=int,
        default=608,
        help="inference size h,w",
    )
    parser.add_argument("--pred_dir", type=str, default=".", help="pred save json dirs")
    parser.add_argument("--map_target", type=float, default=0.56, help="target mAP")
    parser.add_argument("--fps_target", type=float, default=-1.0, help="target fps")

    config = parser.parse_args()
    print("config:", config)
    return config


if __name__ == "__main__":
    config = parse_config()
    main(config)
