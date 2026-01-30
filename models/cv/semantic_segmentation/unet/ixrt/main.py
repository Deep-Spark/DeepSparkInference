#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import argparse

import torch
import numpy as np
from copy import deepcopy

import mmcv
from datasets import build_dataloader, build_dataset
from common import  create_engine_context, get_io_bindings
from tensorrt import Dims
import tensorrt
from cuda import cuda, cudart


def main(config):
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

    if config.test_mode == "FPS":
        torch.cuda.synchronize()
        start_time = time.time()
        for i in range(config.run_loop):
           context.execute_v2(allocations)

        torch.cuda.synchronize()
        end_time = time.time()
        forward_time = end_time - start_time

        fps = config.run_loop * config.bsz / forward_time
        print("FPS : ", fps)
        print(f"\nCheck FPS         Test : {fps}    Target:{config.fps_target}   State : {'Pass' if fps >= config.fps_target else 'Fail'}")
    
    if config.test_mode == "ACC":
        output = np.zeros(outputs[0]["shape"], outputs[0]["dtype"])
        cfg = mmcv.Config.fromfile(config.dataset_cfg)
        cfg.data.test.test_mode = True

        dataset = build_dataset(cfg.data.test)
        loader_cfg = dict(dist=False, shuffle=False)
        loader_cfg.update({
                k: v
                for k, v in cfg.data.items() if k not in [
                    'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
                    'test_dataloader']})
        test_loader_cfg = {
            **loader_cfg,
            'workers_per_gpu': 1,
            'samples_per_gpu': 1,
            'shuffle': False}
        data_loader = build_dataloader(dataset, **test_loader_cfg)
        results = []
        dataset = data_loader.dataset
        counter = 0
        loader_indices = data_loader.batch_sampler

        for batch_indices, data in zip(loader_indices, data_loader):
            img = data["img"][0].numpy()
            img_meta = data["img_metas"][0].data
            
            h_stride, w_stride = cfg.model.test_cfg.stride
            h_crop, w_crop = cfg.model.test_cfg.crop_size
            batch_size, _, h_img, w_img = img.shape
            num_classes = cfg.model.decode_head.num_classes
            h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
            w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
            preds = np.zeros((batch_size, num_classes, h_img, w_img))
            count_mat = np.zeros((batch_size, 1, h_img, w_img))
            for h_idx in range(h_grids):
                for w_idx in range(w_grids):
                    y1 = h_idx * h_stride
                    x1 = w_idx * w_stride
                    y2 = min(y1 + h_crop, h_img)
                    x2 = min(x1 + w_crop, w_img)
                    y1 = max(y2 - h_crop, 0)
                    x1 = max(x2 - w_crop, 0)
                    crop_img = img[:, :, y1:y2, x1:x2]
                    crop_img = np.ascontiguousarray(crop_img)
                    # print(f'crop_img.shape:{crop_img.shape}\n') # 1, 3, 769, 769

                    err, = cuda.cuMemcpyHtoD(inputs[0]["allocation"], crop_img, crop_img.nbytes)
                    assert(err == cuda.CUresult.CUDA_SUCCESS)
                    context.execute_v2(allocations)
                    err, = cuda.cuMemcpyDtoH(output, outputs[0]["allocation"], outputs[0]["nbytes"])
                    assert(err == cuda.CUresult.CUDA_SUCCESS)
                    out= output
                    
                    crop_seg_logit_pad = np.pad(out,
                                    ((0,0),(0,0),
                                    (int(y1), int(preds.shape[2] - y2)),
                                    (int(x1), int(preds.shape[3] - x2)) 
                                    ))
                    preds += crop_seg_logit_pad
                    count_mat[:, :, y1:y2, x1:x2] += 1
        
            preds = preds / count_mat
            preds = np.argmax(preds,axis=1)
            results.extend(preds)
            counter += 1
            print(" [infer] {} / {}".format(counter, len(dataset)), end = "\r")

        eval_kwargs = {"metric": config.metric}
        metric = dataset.evaluate(results, **eval_kwargs)
        print("\n[result] {} : {}\n".format(config.metric, metric[config.metric]))
        metricResult = {"metricResult": {}}
        metricResult["metricResult"][config.metric] = metric[config.metric]
        print(metricResult)

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", type=str, default="float16", help="int8 float16")
    parser.add_argument("--model_type", type=str, default="unet", help="unet ResNet50 Vgg16 MobileNet")
    parser.add_argument("--test_mode", type=str, default="ACC", help="FPS ACC MAP")
    parser.add_argument("--engine_file", type=str, help="engine file path")
    parser.add_argument("--quant_file", type=str, help="weights file path")
    parser.add_argument("--warm_up", type=int, default=-1, help="warm_up times")
    parser.add_argument("--bsz", type=int, default=16, help="test batch size")
    parser.add_argument("--imgh", type=int, default=64, help="inference size h")
    parser.add_argument("--imgw", type=int, default=64, help="inference size w")
    parser.add_argument("--use_async", action="store_true")
    parser.add_argument("--dataset_cfg", type=str, help="datasets file path")
    parser.add_argument("--metric", type=str, help="mIoU mDice")
    parser.add_argument("--device", type=int, default=0, help="cuda device, i.e. 0 or 0,1,2,3,4")
    parser.add_argument("--fps_target", type=float, default=-1.0)
    parser.add_argument("--map_target", type=float, default=-1.0)
    parser.add_argument("--run_loop", type=int, default=-1)

    config = parser.parse_args()
    return config

if __name__ == "__main__":
    config = parse_config()
    try:
        from dltest import show_infer_arguments
        show_infer_arguments(config)
    except:
        pass

    main(config)

