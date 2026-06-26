import os
import cv2
import random
import argparse
import numpy as np
from random import shuffle
from utils import input_transform
from tensorrt.deploy import static_quantize

import torch
import torchvision.datasets
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,  default="ddrnet23.onnx")
    parser.add_argument("--dataset_dir", type=str,  default="/root/data/datasets/cityscapes")
    parser.add_argument("--list_path", type=str,  default="/root/data/datasets/cityscapes/val.lst", help="The path of val list.")
    parser.add_argument("--save_dir", type=str,  help="quant file", default=None)
    args = parser.parse_args()
    return args


def getdataloader(datadir, list_path, step=32, batch_size=4):
    num = step * batch_size
    
    img_list = [line.strip().split()[0] for line in open(list_path)]
    val_list = [os.path.join(datadir, x) for x in img_list]
    random.shuffle(val_list)
    pic_list = val_list[:num]

    dataloader = []
    # imgsz = (1024, 2048)
    for file_path in pic_list:
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        img = input_transform(
            img, 
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
        img = img.transpose((2, 0, 1))
        dataloader.append(img)
    
    calibration_dataset = dataloader
    calibration_dataloader = DataLoader(
        calibration_dataset,
        shuffle=True,
        batch_size=batch_size,
        drop_last=True
    )
    return calibration_dataloader 


args = parse_args()
model_name = os.path.basename(args.model)
model_name = model_name.rsplit(".", maxsplit=1)[0]

out_dir = os.path.dirname(args.model)
dataloader = getdataloader(args.dataset_dir, args.list_path)

static_quantize(args.model,
        calibration_dataloader=dataloader,
        save_quant_onnx_path=os.path.join(out_dir, f"quantized_{model_name}.onnx"),
        save_quant_params_path=os.path.join(out_dir, f"quantized_ddrnet23.json"),
        observer="percentile",
        analyze=True, 
        quant_format="qdq",
        data_preprocess=lambda x: x.to("cuda"),
    )
