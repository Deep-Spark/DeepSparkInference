import os
import cv2
import random
import argparse
import numpy as np
from tensorrt.deploy import static_quantize

import torch
from torch.utils.data import DataLoader
from common import letterbox


def setseed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--model", type=str,  default="yolov4_bs16_without_decoder.onnx")
    parser.add_argument("--dataset_dir", type=str,  default="./coco2017/val2017")
    parser.add_argument("--ann_file", type=str,  default="./coco2017/annotations/instances_val2017.json")
    parser.add_argument("--observer", type=str, choices=["hist_percentile", "percentile", "minmax", "entropy", "ema"], default="hist_percentile")
    parser.add_argument("--disable_quant_names", nargs='*', type=str)
    parser.add_argument("--save_quant_model", type=str,  help="save the quantization model path", default=None)
    parser.add_argument("--bsz", type=int, default=16)
    parser.add_argument("--step", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--imgsz", type=int, default=608)
    parser.add_argument("--use_letterbox", action="store_true")
    args = parser.parse_args()
    return args

args = parse_args()
setseed(args.seed)
model_name = args.model_name


def get_dataloader(data_dir, step=32, batch_size=16, new_shape=[608, 608], use_letterbox=False):
    num = step * batch_size
    val_list = [os.path.join(data_dir, x) for x in os.listdir(data_dir)]
    random.shuffle(val_list)
    pic_list = val_list[:num]

    calibration_dataset = []
    for file_path in pic_list:
        pic_data = cv2.imread(file_path)
        org_img = pic_data
        assert org_img is not None, 'Image not Found ' + file_path
        h0, w0 = org_img.shape[:2]

        if use_letterbox:
            img, ratio, dwdh = letterbox(org_img, new_shape=(new_shape[1], new_shape[0]), auto=False, scaleup=True)
        else:
            img = cv2.resize(org_img, new_shape)
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img) / 255.0  # 0~1 np array
        img = torch.from_numpy(img).float()

        calibration_dataset.append(img)

    calibration_dataloader = DataLoader(
        calibration_dataset,
        shuffle=True,
        batch_size=batch_size,
        drop_last=True
    )
    return calibration_dataloader

dataloader = get_dataloader(
    data_dir=args.dataset_dir,
    step=args.step,
    batch_size=args.bsz,
    new_shape=(args.imgsz, args.imgsz),
    use_letterbox=args.use_letterbox
)

dirname = os.path.dirname(args.save_quant_model)
quant_json_path = os.path.join(dirname, f"quantized_{model_name}.json")

static_quantize(args.model,
        calibration_dataloader=dataloader,
        save_quant_onnx_path=args.save_quant_model,
        save_quant_params_path=quant_json_path,
        observer=args.observer,
        data_preprocess=lambda x: x.to("cuda"),
        quant_format="qdq",
        disable_quant_names=args.disable_quant_names)
