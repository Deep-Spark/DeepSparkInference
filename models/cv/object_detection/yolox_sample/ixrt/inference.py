#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import glob
import json
import os
import time
import sys

import torch
import numpy as np
import cuda.cuda as cuda
import cuda.cudart as cudart

from coco_labels import coco80_to_coco91_class, labels
from common import save2json, box_class85to6
from common import create_engine_context, get_io_bindings
from calibration_dataset import create_dataloaders
from datasets.post_process import get_post_process

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
from tqdm.contrib import tzip

import tensorrt

from load_ixrt_plugin import load_ixrt_plugin
load_ixrt_plugin()

def main(config):

    # Load dataloader
    dataloader = create_dataloaders(
        data_path=config.eval_dir,
        annFile=config.coco_gt,
        img_sz=config.imgsz,
        batch_size=config.bsz,
        step=config.loop_count,
        data_process_type=config.data_process_type
    )

    # Load post process func
    if config.test_mode == "MAP":
        post_process_func = get_post_process(config.data_process_type)

    bsz = config.bsz
    num_samples = 5000
    if config.loop_count > 0:
        num_samples = bsz * config.loop_count
    num_batch = len(dataloader)
    print("=" * 30)
    print(f"Test Mode : {'Asynchronous' if config.use_async else 'Synchronous'}")
    print(f"Total sample : {num_samples}\nBatch_size : {bsz}\nRun Batch : {num_batch}")
    print("=" * 30)

    json_result = []
    forward_time = 0.0
    class_map = coco80_to_coco91_class()

    host_mem = tensorrt.IHostMemory
    logger = tensorrt.Logger(tensorrt.Logger.ERROR)

    # Load Engine
    engine, context = create_engine_context(config.model_engine, logger)
    inputs, outputs, allocations = get_io_bindings(engine)

    # Load nms_engine
    if config.test_mode == "MAP" and config.nms_type == "GPU":
        nms_engine, nms_context = create_engine_context(config.nms_engine, logger)
        nms_inputs, nms_outputs, nms_allocations = get_io_bindings(nms_engine)
        nms_output0 = np.zeros(nms_outputs[0]["shape"], nms_outputs[0]["dtype"])
        nms_output1 = np.zeros(nms_outputs[1]["shape"], nms_outputs[1]["dtype"])
        print(f"nms_output0 shape : {nms_output0.shape}   nms_output0 type : {nms_output0.dtype}")
        print(f"nms_output1 shape : {nms_output1.shape}   nms_output1 type : {nms_output1.dtype}")

    # Warm up
    if config.warm_up > 0:
        print("\nWarm Start.")
        for i in range(config.warm_up):
            context.execute_v2(allocations)
        print("Warm Done.")

    # Prepare the output data
    output = np.zeros(outputs[0]["shape"], outputs[0]["dtype"])
    print(f"output shape : {output.shape} output type : {output.dtype}")

    for batch_data, batch_img_shape, batch_img_id in tqdm(dataloader):
        batch_data = batch_data.numpy()
        batch_img_shape = [batch_img_shape[0].numpy(), batch_img_shape[1].numpy()]
        # batch_img_id = batch_img_id.numpy()

        cur_bsz_sample = batch_data.shape[0]

        # Set input
        err, = cuda.cuMemcpyHtoD(inputs[0]["allocation"], batch_data, batch_data.nbytes)
        assert(err == cuda.CUresult.CUDA_SUCCESS)

        # Forward
        # start_time = time.time()
        context.execute_v2(allocations)
        # end_time = time.time()
        # forward_time += end_time - start_time

        if config.test_mode == "MAP":
            # Fetch output
            err, = cuda.cuMemcpyDtoH(output, outputs[0]["allocation"], outputs[0]["nbytes"])
            assert(err == cuda.CUresult.CUDA_SUCCESS)


            # Step 1 : prepare data to nms
            _, box_num, box_unit = output.shape
            if config.debug:
                print(f"[Debug] box_num(25200) : {box_num}, box_unit(6) : {box_unit}")

            if config.decoder_faster == 0:
                nms_input = box_class85to6(output.reshape(-1, box_unit))
            else:
                nms_input = output

            # Step 2 : nms
            # cpu nms(TODO)

            # gpu nms
            if config.nms_type == "GPU":

                # Set nms input
                err, = cuda.cuMemcpyHtoD(nms_inputs[0]["allocation"], nms_input, nms_input.nbytes)
                assert(err == cuda.CUresult.CUDA_SUCCESS)
                nms_context.execute_v2(nms_allocations)
                err, = cuda.cuMemcpyDtoH(nms_output0, nms_outputs[0]["allocation"], nms_outputs[0]["nbytes"])
                assert(err == cuda.CUresult.CUDA_SUCCESS)
                err, = cuda.cuMemcpyDtoH(nms_output1, nms_outputs[1]["allocation"], nms_outputs[1]["nbytes"])
                assert(err == cuda.CUresult.CUDA_SUCCESS)

            # Step 3 : post process + save
            pred_boxes = post_process_func(
                ori_img_shape=batch_img_shape,
                imgsz=(config.imgsz, config.imgsz),
                box_datas=nms_output0,
                box_nums=nms_output1,
                sample_num=cur_bsz_sample,
                max_det=config.max_det
            )
            save2json(batch_img_id, pred_boxes, json_result, class_map)

    # fps = num_samples / forward_time

    if config.test_mode == "FPS":
        start_time = time.time()       
        for i in range(config.loop_count):
            context.execute_v2(allocations)  
        end_time = time.time()  
        forward_time = end_time - start_time      
        fps = (config.loop_count*config.bsz) / forward_time
        print("FPS : ", fps)
        print(f"Performance Check : Test {fps} >= target {config.fps_target}")
        if fps >= config.fps_target:
            print("pass!")
            exit()
        else:
            print("failed!")
            exit(10)

    if config.test_mode == "MAP":
        if len(json_result) == 0:
            print("Predict zero box!")
            exit(10)

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
        if map50 >= config.map_target:
            print("pass!")
            exit()
        else:
            print("failed!")
            exit(10)

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default="YOLOV5s", help="YOLOV3 YOLOV5 YOLOV7 YOLOX"
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
        "--nms_engine",
        type=str,
        default="",
        help="nms engine path",
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
        default=640,
        help="inference size h,w",
    )
    parser.add_argument("--max_det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--data_process_type", type=str,  default="none")
    parser.add_argument("--use_async", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--pred_dir", type=str, default=".", help="pred save json dirs")
    parser.add_argument("--map_target", type=float, default=0.56, help="target mAP")
    parser.add_argument("--fps_target", type=float, default=-1.0, help="target fps")
    parser.add_argument("--decoder_faster", type=int, default=0, help="decoder faster can use gpu nms directly")
    parser.add_argument("--nms_type", type=str, default="GPU", help="GPU/CPU")

    config = parser.parse_args()
    print("config:", config)
    return config

if __name__ == "__main__":
    config = parse_config()
    main(config)