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
import ctypes
import math
import os
import sys
import time
import cv2
import json
import numpy as np
from tqdm import tqdm
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from backbone import build_resnet50
from head import add_roi_heads, add_rpn
from img_postprocess import postprocess_img, postprocess_img_and_save
from img_preprocess import preprocess_img
from utils import load_weights
from dataloader import create_dataloaders, coco80_to_coco91_class

INPUT_H = 800
INPUT_W = 1067
INPUT_NODE_NAME = "images"
RES2_OUT_CHANNELS = 256


def save2json(batch_img_id, pred_boxes, json_result, class_trans):
    if len(pred_boxes) > 0:
        image_id = int(batch_img_id)
        # have no target
        if image_id == -1:
            return
        for x, y, w, h, c, p, mask in pred_boxes:
            x, y, w, h, p = float(x), float(y), float(w), float(h), float(p)
            c = int(c)
            json_result.append(
                    {
                        "image_id": image_id,
                        "category_id": class_trans[c],
                        "bbox": [x, y, w, h],
                        "score": p,
                        "segmentation": mask,
                    }
                )


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
        size = np.dtype(trt.nptype(dtype)).itemsize
        for s in shape:
            size *= s
        allocation = cuda.mem_alloc(size)
        binding = {
            "index": i,
            "name": name,
            "dtype": np.dtype(trt.nptype(dtype)),
            "shape": list(shape),
            "allocation": allocation,
        }
        allocations.append(allocation)
        if engine.binding_is_input(i):
            inputs.append(binding)
        else:
            outputs.append(binding)
    return inputs, outputs, allocations


def build_rcnn_model(wtsfile, engine_file):
    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)

    build_config = builder.create_builder_config()
    build_config.set_flag(trt.BuilderFlag.FP16)

    data = network.add_input(
        name=INPUT_NODE_NAME, dtype=trt.float32, shape=[1, 3, INPUT_H, INPUT_W]
    )
    weight_map = load_weights(wtsfile)

    features = build_resnet50(network, weight_map, data, 64, 64, RES2_OUT_CHANNELS)
    proposals = add_rpn(network, weight_map, features)
    results = add_roi_heads(network, weight_map, proposals, features)

    for result in results:
        network.mark_output(result)
    plan = builder.build_serialized_network(network, build_config)

    with open(engine_file, "wb") as f:
        f.write(plan)

    print("Build engine done!")


def run_maskrcnn(engine_file, image_folder):
    cuda.init()
    logger = trt.Logger(trt.Logger.VERBOSE)
    engine_file_buffer = open(engine_file, "rb")
    runtime = trt.Runtime(logger)
    assert runtime
    engine = runtime.deserialize_cuda_engine(engine_file_buffer.read())
    assert engine
    context = engine.create_execution_context()
    assert context

    # Setup I/O bindings
    inputs, outputs, allocations = setup_io_bindings(engine, context)

    ### infer
    # Prepare the output data
    output = np.zeros(outputs[0]["shape"], outputs[0]["dtype"])
    for image_name in os.listdir(image_folder):
        src_path = os.path.join(image_folder, image_name)
        dst_path = "_" + image_name
        img = cv2.imread(src_path)
        _, h_ori, w_ori = img.shape
        out, pad_list = preprocess_img(img, INPUT_W, INPUT_H)

        data_batch = out.reshape(1, *out.shape)
        data_batch = np.ascontiguousarray(data_batch)

        # Process I/O and execute the network
        cuda.memcpy_htod(inputs[0]["allocation"], data_batch)
        context.execute_v2(allocations)
        output_h_list = []
        for output in outputs:
            output_np = np.zeros(output["shape"], output["dtype"])
            cuda.memcpy_dtoh(output_np, output["allocation"])
            output_h_list.append(output_np)
        scores_h = output_h_list[0]
        boxes_h = output_h_list[1]
        classes_h = output_h_list[2]
        print(scores_h)
        print(boxes_h)
        masks_h = output_h_list[3]
        postprocess_img_and_save(
            img,
            INPUT_W,
            INPUT_H,
            scores_h[0],
            boxes_h[0],
            classes_h[0],
            masks_h[0],
            pad_list,
            dst_path,
        )

    # Free Gpu Memory
    inputs[0]["allocation"].free()
    for output in outputs:
        output["allocation"].free()
    engine_file_buffer.close()


def get_maskrcnn_perf(config):
    cuda.init()
    logger = trt.Logger(trt.Logger.WARNING)
    engine_file_buffer = open(config.engine_file, "rb")
    runtime = trt.Runtime(logger)
    assert runtime
    engine = runtime.deserialize_cuda_engine(engine_file_buffer.read())
    assert engine
    context = engine.create_execution_context()
    assert context

    # Setup I/O bindings
    inputs, outputs, allocations = setup_io_bindings(engine, context)

    ### infer
    # Prepare the output data
    output = np.zeros(outputs[0]["shape"], outputs[0]["dtype"])

    data_batch = np.zeros((1, 3, INPUT_H, INPUT_W), dtype=np.float32)
    data_batch = np.ascontiguousarray(data_batch)

    # Process I/O and execute the network
    cuda.memcpy_htod(inputs[0]["allocation"], data_batch)
    # warm up 
    print("Warmup start ...")
    for i in range(5):
        context.execute_v2(allocations)
    print("Warmup done !\nStart forward ...")
    # run 
    forward_time = 0
    for i in range(20):
        start_time = time.time()
        context.execute_v2(allocations)
        end_time = time.time()
        forward_time += end_time - start_time
    fps = 20.0 / forward_time
    print("Forward done !")

    # Free Gpu Memory
    inputs[0]["allocation"].free()
    for output in outputs:
        output["allocation"].free()
    engine_file_buffer.close()

    print("\nFPS : ", fps)
    print(f"Performance Check : Test {fps} >= target {config.fps_target}")
    if fps >= config.fps_target:
        print("pass!")
    else:
        print("failed!")
        exit(1)

def get_maskrcnn_acc(config):
    json_result = []
    class_map = coco80_to_coco91_class()

    # Load dataloader
    anno_json = os.path.join(config.dataset_dir, "annotations/instances_val2017.json")
    imgs_dir = os.path.join(config.dataset_dir, "images/val2017")
    dataloader = create_dataloaders(
        data_path=imgs_dir,
        annFile=anno_json,
        img_h=INPUT_H,
        img_w=INPUT_W,
        batch_size=config.bsz,
    )
    print("COCO Val2017 Datasets images : ", len(dataloader))

    cuda.init()
    logger = trt.Logger(trt.Logger.WARNING)
    engine_file_buffer = open(config.engine_file, "rb")
    runtime = trt.Runtime(logger)
    assert runtime
    engine = runtime.deserialize_cuda_engine(engine_file_buffer.read())
    assert engine
    context = engine.create_execution_context()
    assert context

    # Setup I/O bindings
    inputs, outputs, allocations = setup_io_bindings(engine, context)

    ### infer
    # Prepare the output data
    output = np.zeros(outputs[0]["shape"], outputs[0]["dtype"])

    # warm up 
    print("Warmup start ...")
    for i in range(3):
        context.execute_v2(allocations)
    print("Warmup done !\nStart forward ...")
    
    # run
    for batch_data, batch_img_shape, batch_img_id, batched_paddings, paths in tqdm(dataloader):
        batch_data = batch_data.numpy()
        batch_img_shape = batch_img_shape.numpy()
        batch_img_id = batch_img_id.numpy()
        batched_paddings = batched_paddings.numpy()
        img_path = paths[0]
        img = cv2.imread(img_path, 1)
        # Process I/O and execute the network
        # cpu -> gpu
        batch_data = np.ascontiguousarray(batch_data)
        cuda.memcpy_htod(inputs[0]["allocation"], batch_data)
        
        context.execute_v2(allocations)

        # gpu -> cpu
        output_h_list = []
        for output in outputs:
            output_np = np.zeros(output["shape"], output["dtype"])
            cuda.memcpy_dtoh(output_np, output["allocation"])
            output_h_list.append(output_np)
        scores_h = output_h_list[0]
        boxes_h = output_h_list[1]
        classes_h = output_h_list[2]
        masks_h = output_h_list[3]

        bboxs_masks = postprocess_img(
            batch_img_shape[0][1], # w
            batch_img_shape[0][0], # h
            INPUT_W,
            INPUT_H,
            scores_h[0],
            boxes_h[0],
            classes_h[0],
            masks_h[0],
            batched_paddings[0]
        )
        save2json(batch_img_id, bboxs_masks, json_result, class_map)
        
    print("Forward done !")
    
    tmp_result_name = "pred_results.json"
    if os.path.exists(tmp_result_name):
        os.remove(tmp_result_name)
    with open(tmp_result_name, "w") as f:
        json.dump(json_result, f)

    
    anno = COCO(anno_json)  # init annotations api
    pred = anno.loadRes(tmp_result_name)  # init predictions api
    
    eval = COCOeval(anno, pred, "bbox")
    eval.evaluate()
    eval.accumulate()
    print(f"==============================eval COCO bbox mAP ==============================")
    eval.summarize()

    segm_eval = COCOeval(anno, pred, "segm")
    segm_eval.evaluate()
    segm_eval.accumulate()
    print(f"==============================eval COCO segm mAP ==============================")
    segm_eval.summarize()

    _, map50 = eval.stats[:2]
    print("bbox mAP@0.5 : ", map50)
    print(f"bbox Accuracy Check : Test {map50} >= target {config.map_target}")
    
    _, segm_map50 = segm_eval.stats[:2]
    print("segm mAP@0.5 : ", segm_map50)
    print(f"segm Accuracy Check : Test {segm_map50} >= target {config.segm_map_target}")
    
    if map50 >= config.map_target and segm_map50 >= config.segm_map_target:
        print("pass!")
    else:
        print("failed!")
        exit(1)

    # Free Gpu Memory
    inputs[0]["allocation"].free()
    for output in outputs:
        output["allocation"].free()
    engine_file_buffer.close()


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["build_engine", "run_engine", "perf", "acc"])
    parser.add_argument(
        "--wts_file",
        type=str,
        default=None,
        help="wts file path",
    )
    parser.add_argument(
        "--engine_file",
        type=str,
        default=None,
        required=True,
        help="engine file path",
    )
    parser.add_argument(
        "--test_folder",
        type=str,
        default=None,
        help="test images folder",
    )
    parser.add_argument(
        "--bsz",
        type=int,
        default=1,
        help="support bsz = 1 only",
    )
    parser.add_argument(
        "--img_h",
        type=int,
        default=1,
        help="net input height",
    )
    parser.add_argument(
        "--img_w",
        type=int,
        default=1,
        help="net input width",
    )
    parser.add_argument(
        "--fps_target",
        type=float,
        default=7.2,
        help="fps target setting mr50",
    )
    parser.add_argument(
        "--map_target",
        type=float,
        default=0.4,
        help="bbox map 0.5 target setting",
    )
    parser.add_argument(
        "--segm_map_target",
        type=float,
        default=0.4,
        help="sgem map 0.5 target setting",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="dataset coco dir",
    )
    config = parser.parse_args()
    return config


if __name__ == "__main__":
    config = parse_config()
    if config.action == "build_engine":
        if config.wts_file is None:
            print("build engine : set --wts_file first .")
            sys.exit()
        build_rcnn_model(config.wts_file, config.engine_file)
    elif config.action == "run_engine":
        if config.test_folder is None:
            sys.exit()
        run_maskrcnn(config.engine_file, config.test_folder)
    elif config.action == "perf":
        if config.engine_file is None:
            print("run performance : set --engine_file first .")
            sys.exit()
        if config.bsz != 1:
            print("run performance : set --bsz 1  first .")
            sys.exit()
        get_maskrcnn_perf(config)
    elif config.action == "acc":
        if config.engine_file is None:
            print("run acc : set --engine_file first .")
            sys.exit()
        if config.dataset_dir is None:
            print("run acc : set --dataset_dir first .")
            sys.exit()
        if config.bsz != 1:
            print("run acc : set --bsz 1  first .")
            sys.exit()
        get_maskrcnn_acc(config)
