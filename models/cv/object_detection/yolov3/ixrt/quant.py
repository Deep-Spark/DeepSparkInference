import os
import random
import argparse
import numpy as np
from tensorrt.deploy import static_quantize

import torch
from calibration_dataset import create_dataloaders

def setseed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--model", type=str,  default="yolov5s_with_decoder.onnx")
    parser.add_argument("--data_process_type", type=str,  default="none")
    parser.add_argument("--dataset_dir", type=str,  default="./coco2017/val2017")
    parser.add_argument("--ann_file", type=str,  default="./coco2017/annotations/instances_val2017.json")
    parser.add_argument("--observer", type=str, choices=["hist_percentile", "percentile", "minmax", "entropy", "ema"], default="hist_percentile")
    parser.add_argument("--disable_quant_names", nargs='*', type=str)
    parser.add_argument("--save_dir", type=str,  help="save path", default=None)
    parser.add_argument("--bsz", type=int, default=32)
    parser.add_argument("--step", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    return args

args = parse_args()
setseed(args.seed)
model_name = args.model_name

out_dir = args.save_dir
dataloader = create_dataloaders(
    data_path=args.dataset_dir,
    annFile=args.ann_file,
    img_sz=args.imgsz,
    batch_size=args.bsz,
    step=args.step,
    data_process_type=args.data_process_type
)
# print("disable_quant_names : ", args.disable_quant_names)
static_quantize(args.model,
        calibration_dataloader=dataloader,
        save_quant_onnx_path=os.path.join(out_dir, f"quantized_{model_name}.onnx"),
        observer=args.observer,
        data_preprocess=lambda x: x[0].to("cuda"),
        quant_format="qdq",
        disable_quant_names=args.disable_quant_names)