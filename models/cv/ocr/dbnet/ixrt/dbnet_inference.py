#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import argparse
import sys


from cuda import cuda, cudart
import torch
import tensorrt

from util.common import eval_batch, create_engine_context, get_io_bindings

from util import TextDetector

def check_target(inference, target):
    satisfied = False
    if inference > target:
        satisfied = True  
    return satisfied      

process_configs ={
    #pre process config 
    'std': [0.229, 0.224, 0.225],
    'mean': [0.485, 0.456, 0.406],
    'scale': 1./255.,
    'image_shape':(1280,736),#width height
    
    #post precess config
    'thresh':0.3,
    'box_thresh':0.5,
    'max_candidates':1000,
    'unclip_ratio':2,
    'use_dilation':False,
    'score_mode':'fast',
    'box_type':'quad',
    'batch_size':1

}

def make_parser():
    parser = argparse.ArgumentParser("DBnet Eval")
    parser.add_argument("--datasets_dir", type=str, default="data/icdar_2015_images",  help="datasets dir ")
    parser.add_argument("--engine_file", type=str, default="data/unit_test_r50_en_dbnet_bin/int8_r50_en_dbnet.engine",  help="weights dir")

    parser.add_argument("-b", "--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("-d", "--device", default=1, type=int, help="device for val")
    parser.add_argument("--img_height", default=736, type=int, help="test img height")
    parser.add_argument("--img_width", default=1280, type=int, help="test img width")
    parser.add_argument("--target_hmean", default=0.82, type=float, help="target Hmean")
    parser.add_argument("--target_fps", default=30, type=float, help="target Hmean")
    
    parser.add_argument("--target", default="precision", type=str, help="precision or pref")
    parser.add_argument("--warm_up", default=20, type=int , help="warm_up")
    parser.add_argument("--loop_count", default=100, type=int , help="loop_count")
    parser.add_argument("--seed", default=None, type=int, help="eval seed")
    return parser




def eval(args):  
    
    host_mem = tensorrt.IHostMemory
    logger = tensorrt.Logger(tensorrt.Logger.ERROR)
    engine, context = create_engine_context(args.engine_file, logger)  
    
    process_configs["image_dir"]=args.datasets_dir
    process_configs["label_dir"] = args.datasets_dir
    process_configs["image_shape"]= (args.img_height,args.img_width)
    process_configs["batch_size"] = args.batch_size
    db_det = TextDetector(engine,context,process_configs)
    if args.target=="precision":
        metrics = db_det.eval_icdar_2015(args.datasets_dir,args.batch_size)
        print("="*40)
        print("Precision:{0},Recall:{1},Hmean:{2}".format(round(metrics["precision"],3),round(metrics["recall"],3),round(metrics["hmean"],3)))
        print("="*40)
        print(f"Check hmean Test : {round(metrics['hmean'],3)}  Target:{args.target_hmean} \
              State : {'Pass' if round(metrics['hmean'],3) >= args.target_hmean else 'Fail'}")
        status_hmean = check_target(metrics["hmean"], args.target_hmean)
        metricResult = {"metricResult": {}}
        metricResult["metricResult"]["hmean"] = round(metrics["hmean"], 3)
        print(metricResult)
        sys.exit(int(not (status_hmean)))
    else:
        fps = db_det.perf(args.warm_up,args.loop_count,args.batch_size)
        print("="*40)
        print("fps:{0}".format(round(fps,2)))
        print("="*40)
        print(f"Check fps Test : {round(fps,3)}  Target:{args.target_fps} State : {'Pass' if  fps >= args.target_fps else 'Fail'}")
        status_fps = check_target(fps, args.target_fps)
        metricResult = {"metricResult": {}}
        metricResult["metricResult"]["fps"] = round(fps, 3)
        print(metricResult)
        sys.exit(int(not (status_fps)))
            
if __name__ == "__main__":
    args = make_parser().parse_args()
    eval(args)
    
        
    

