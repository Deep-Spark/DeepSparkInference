import os
import cv2
import random
import argparse
import numpy as np
from random import shuffle
from tensorrt.deploy import static_quantize

import torch
import torchvision.datasets
from calibration_dataset import getdataloader

def setseed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--dataset_dir", type=str, default="imagenet_val")
    parser.add_argument("--observer", type=str, choices=["hist_percentile", "percentile", "minmax", "entropy", "ema"], default="hist_percentile")
    parser.add_argument("--disable_quant_names", nargs='*', type=str)
    parser.add_argument("--save_dir", type=str,  help="save path", default=None)
    parser.add_argument("--bsz", type=int, default=32)
    parser.add_argument("--step", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--imgsz", type=int, default=224)
    args = parser.parse_args()
    print("Quant config:", args)
    print(args.disable_quant_names)
    return args

args = parse_args()
setseed(args.seed)
calibration_dataloader = getdataloader(args.dataset_dir, args.step, args.bsz, img_sz=args.imgsz)
static_quantize(args.model,
        calibration_dataloader=calibration_dataloader,
        save_quant_onnx_path=os.path.join(args.save_dir, f"quantized_{args.model_name}.onnx"),
        observer=args.observer,
        data_preprocess=lambda x: x[0].to("cuda"),
        quant_format="qdq",
        disable_quant_names=args.disable_quant_names)