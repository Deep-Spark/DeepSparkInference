#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
import tensorrt
from tensorrt.utils import topk
from sklearn import metrics
from scipy.optimize import brentq
from sklearn.model_selection import KFold
from scipy import interpolate

from utils import read_pairs, get_paths, evaluate
from common import getdataloader, create_engine_context, get_io_bindings
from load_ixrt_plugin import load_ixrt_plugin
load_ixrt_plugin()

def main(config):
    embed_loader, crop_paths = getdataloader(config.datasets_dir, config.loop_count, config.bsz, config.imgsz)

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

        classes = []
        embeddings = []

        for xb, yb in tqdm(embed_loader):
        
            output = np.zeros(outputs[0]["shape"], outputs[0]["dtype"])
            current_imgs_num = xb.numpy().shape[0]
            xb = xb.numpy()
            xb = np.ascontiguousarray(xb)

            err, = cuda.cuMemcpyHtoD(inputs[0]["allocation"], xb, xb.nbytes)
            assert(err == cuda.CUresult.CUDA_SUCCESS)
            context.execute_v2(allocations)
            err, = cuda.cuMemcpyDtoH(output, outputs[0]["allocation"], outputs[0]["nbytes"])
            assert(err == cuda.CUresult.CUDA_SUCCESS)

            output = output.reshape(output.shape[0],output.shape[1])
            #print("output shape ",output.shape)

            classes.extend(yb[0:current_imgs_num].numpy())
            embeddings.extend(output)


        embeddings_dict = dict(zip(crop_paths,embeddings))

        pairs = read_pairs(config.datasets_dir + config.pairs_name)
        path_list, issame_list = get_paths(config.datasets_dir + 'lfw', pairs)
        # embeddings = np.array([embeddings_dict[path.replace(".png",".jpg")] for path in path_list])
        embeddings = np.array([embeddings_dict[path] for path in path_list])
        tpr, fpr, accuracy, val, val_std, far, fp, fn = evaluate(embeddings, issame_list)

        print('\nAccuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
        print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))

        auc = metrics.auc(fpr, tpr)
        print('Area Under Curve (AUC): %1.3f' % auc)
        #eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr, fill_value="extrapolate")(x), 0., 1.)
        #print('Equal Error Rate (EER): %1.3f' % eer)

        acc = np.mean(accuracy)
        print(f"Accuracy Check : Test {acc} >= target {config.acc_target}")
        if acc >= config.acc_target:
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
    parser.add_argument("--pairs_name", type=str, default="pairs.txt", help="binary weights file name")
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
