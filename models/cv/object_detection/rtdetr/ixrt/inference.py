#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import time
import numpy as np
from cuda import cuda, cudart

from coco_labels import coco80_to_coco91_class, labels
from common import save2json
from common import create_engine_context
from calibration_dataset import create_dataloaders
from datasets.post_process import get_post_process
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
from tqdm.contrib import tzip

import tensorrt

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
    print(f"Total sample : {num_samples}\nBatch_size : {bsz}\nRun Batch : {num_batch}")
    print("=" * 30)

    json_result = []
    forward_time = 0.0
    class_map = coco80_to_coco91_class()

    host_mem = tensorrt.IHostMemory
    logger = tensorrt.Logger(tensorrt.Logger.ERROR)

    # Load Engine
    engine, context = create_engine_context(config.model_engine, logger)

    # Setup I/O bindings
    inputs = []
    outputs = []
    for i in range(engine.num_io_tensors):
        tensor_name = engine.get_tensor_name(i)
        dtype = engine.get_tensor_dtype(tensor_name)
        shape = engine.get_tensor_shape(tensor_name)
        size = np.dtype(tensorrt.nptype(dtype)).itemsize
        np_dtype = np.dtype(tensorrt.nptype(dtype))
        if -1 in list(shape):
            if engine.get_tensor_mode(tensor_name) == tensorrt.TensorIOMode.INPUT:
                shape = engine.get_tensor_profile_shape(tensor_name, 0)[2]
                context.set_input_shape(tensor_name, shape)
                for s in shape:
                    size *= s
            else:
                shape = context.get_tensor_shape(tensor_name)
                for s in shape:
                    size *= s
        else:
            for s in shape:
                size *= s

        err, allocation = cudart.cudaMalloc(size)
        assert err == cudart.cudaError_t.cudaSuccess
        context.set_tensor_address(tensor_name, int(allocation))
        binding = {
            "name": tensor_name,
            "dtype": np.dtype(tensorrt.nptype(dtype)),
            "allocation": allocation,
        }
        if engine.get_tensor_mode(tensor_name) == tensorrt.TensorIOMode.INPUT:
            inputs.append(binding)
        else:
            outputs.append(binding)

    err, stream = cudart.cudaStreamCreate()
    assert err == cudart.cudaError_t.cudaSuccess

    # Warm up
    print(config.warm_up)
    if config.warm_up > 0:
        print("\nWarm Start.")
        for i in range(config.warm_up):
            context.execute_async_v3(stream)
        print("Warm Done.")

    # Prepare the output data
    for batch_data, batch_img_shape, batch_img_id in tqdm(dataloader):
        batch_data = batch_data.numpy()
        batch_img_shape = [batch_img_shape[0].numpy(), batch_img_shape[1].numpy()]
        cur_bsz_sample = batch_data.shape[0]  
        # Set input
        context.set_input_shape(inputs[0]["name"], batch_data.shape)
        err, = cudart.cudaMemcpy(inputs[0]["allocation"], batch_data, batch_data.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        assert err == cudart.cudaError_t.cudaSuccess

        # Forward
        start_time = time.time()
        context.execute_async_v3(stream)
        end_time = time.time()
        forward_time += end_time - start_time

        if config.test_mode == "MAP":
            # Fetch output
            output_shape = context.get_tensor_shape(outputs[0]["name"])
            model_output = np.zeros(output_shape, outputs[0]["dtype"])
            output_size = outputs[0]["dtype"].itemsize
            for s in output_shape:
                output_size *= s
            err, = cudart.cudaMemcpy(model_output, outputs[0]["allocation"], output_size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
            assert err == cudart.cudaError_t.cudaSuccess

            pred_boxes = post_process_func(
                ori_img_shape=batch_img_shape,
                imgsz=(config.imgsz, config.imgsz),
                box_datas=model_output,
                box_num=output_shape[1],
                sample_num=cur_bsz_sample,
                max_det=300
            )
            save2json(batch_img_id, pred_boxes, json_result,class_map)
                

    fps = num_samples / forward_time
    # Free
    for i in range(engine.num_io_tensors):
        tensor_name = engine.get_tensor_name(i)
        (err,) = cudart.cudaFree(context.get_tensor_address(tensor_name))
        assert err == cudart.cudaError_t.cudaSuccess

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
            config.pred_dir, f"rt-detr-v3_{config.precision}_preds.json"
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
            f"==============================eval rt-detr-v3 {config.precision} coco map =============================="
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
            exit(1)

def parse_config():
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--warm_up", type=int, default=0, help="warm_up count")
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
    parser.add_argument("--data_process_type", type=str,  default="none")
    parser.add_argument("--pred_dir", type=str, default=".", help="pred save json dirs")
    parser.add_argument("--map_target", type=float, default=0.56, help="target mAP")
    parser.add_argument("--fps_target", type=float, default=-1.0, help="target fps")

    config = parser.parse_args()
    print("config:", config)
    return config

if __name__ == "__main__":
    config = parse_config()
    main(config)