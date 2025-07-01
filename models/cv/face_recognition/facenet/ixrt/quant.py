import os

import torch
from tensorrt.deploy.api import *
from tensorrt.deploy.utils.seed import manual_seed
from torchvision import models
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms
import json
import cv2
import numpy as np
import math
import simplejson as json
from tensorrt.deploy import static_quantize


# manual_seed(43)
device = 0 if torch.cuda.is_available() else "cpu"


def fixed_image_standardization(image_tensor):
    processed_tensor = (image_tensor - 127.5) / 128.0
    return processed_tensor

def create_dataloader(args):
    image_dir_path = os.path.join(args.data_path, "lfw")

    trans = transforms.Compose([
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization
    ])
    
    dataset = datasets.ImageFolder(args.data_path + 'lfw', transform=trans)

    calibration_dataset = dataset
    print("image folder total images : ", len(dataset))
    if args.num_samples is not None:
        indices = np.random.permutation(len(dataset))[:args.num_samples]
        calibration_dataset = torch.utils.data.Subset(
            dataset, indices=indices
        )
        print("calibration_dataset images : ", len(calibration_dataset))

    assert len(dataset), f"data size is 0, check data path please"
    calibration_dataloader = DataLoader(
        calibration_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )
    verify_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )

    return calibration_dataloader, verify_dataloader


@torch.no_grad()
def quantize_model(args, model_name, model, dataloader):

    calibration_dataloader, verify_dataloader = dataloader
    print("calibration dataset length: ", len(calibration_dataloader))

    if isinstance(model, torch.nn.Module):
        model = model.to(device)
        model.eval()

    static_quantize(args.model,
        calibration_dataloader=calibration_dataloader,
        save_quant_onnx_path=os.path.join("./facenet_weights", f"{model_name}-quant.onnx"),
        observer=args.observer,
        data_preprocess=lambda x: x[0].to("cuda"),
        quant_format="qdq",
        disable_quant_names=None)

def add_1190_scale(cfg_name):
    graph_json = json.load(open(cfg_name))

    graph_json["quant_info"]["1190"] = graph_json["quant_info"]["1189"]

    with open(cfg_name, "w") as fh:
        json.dump(graph_json, fh, indent=4)

def create_argparser(*args, **kwargs):
    parser = ArgumentParser(*args, **kwargs)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--img_size", type=int, default=160)
    parser.add_argument("-j", "--workers", type=int, default=4)
    parser.add_argument("--model", type=str, default="./facenet_weights/facenet.onnx")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--data_path", type=str, default="./facenet_datasets/")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--observer", type=str, default="hist_percentile")
    parser.add_argument("--fp32_acc", action="store_true")
    parser.add_argument("--use_ixrt", action="store_true")
    parser.add_argument("--quant_params", type=str, default=None)
    parser.add_argument("--disable_bias_correction", action="store_true")
    return parser

def parse_args():
    parser = create_argparser("PTQ Quantization")
    args = parser.parse_args()
    args.use_ixquant = not args.use_ixrt
    return args


def main():
    args = parse_args()
    print(args)
    dataloader = create_dataloader(args)

    if args.model.endswith(".onnx"):
        model_name = os.path.basename(args.model)
        model_name = model_name.rsplit(".", maxsplit=1)[0]
        model = args.model
    else:
        print("[Error] file name not correct ", args.model)
    quantize_model(args, model_name, model, dataloader)
    json_name = f"./facenet_weights/{model_name}.json"
    add_1190_scale(json_name)

if __name__ == "__main__":
    main()
