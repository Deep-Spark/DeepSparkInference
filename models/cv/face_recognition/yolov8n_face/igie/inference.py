# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
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

import os
import argparse
import tvm
import cv2
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--engine",
                        type=str,
                        required=True,
                        help="igie engine path.")

    parser.add_argument("--batchsize",
                        type=int,
                        required=True,
                        help="inference batch size.")

    parser.add_argument("--datasets",
                        type=str,
                        required=True,
                        help="datasets path.")

    parser.add_argument("--input_name",
                        type=str,
                        required=True,
                        help="input name of the model.")

    parser.add_argument("--imgsz",
                        type=int,
                        default=640)
    parser.add_argument("--warmup",
                        type=int,
                        default=3,
                        help="number of warmup before test.")

    parser.add_argument("--num_workers",
                        type=int,
                        default=4,
                        help="number of workers used in pytorch dataloader.")

    parser.add_argument("--acc_target",
                        type=float,
                        default=None,
                        help="Model inference Accuracy target.")

    parser.add_argument("--fps_target",
                        type=float,
                        default=None,
                        help="Model inference FPS target.")

    parser.add_argument("--conf",
                        type=float,
                        default=0.01,
                        help="confidence threshold.")

    parser.add_argument("--iou",
                        type=float,
                        default=0.5,
                        help="iou threshold.")

    parser.add_argument("--max_det",
                        type=int,
                        default=2048,
                        help="maximum detections per image.")

    parser.add_argument("--perf_only",
                        type=bool,
                        default=False,
                        help="Run performance test only")

    args = parser.parse_args()

    return args


def letterbox_image(im, imgsz=640):
    """Ultralytics LetterBox with auto=False (fixed 640x640 for exported ONNX/IGIE)."""
    transform = LetterBox(new_shape=(imgsz, imgsz), auto=False, stride=32)
    return transform(image=im)


def resolve_val_list(datasets):
    candidates = [
        os.path.join(datasets, "val", "wider_val.txt"),
        os.path.join(datasets, "wider_val.txt"),
        os.path.join(datasets, "val", "images", "..", "wider_val.txt"),
    ]
    for path in candidates:
        path = os.path.normpath(path)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "WIDER FACE list not found. Expected ${DATASETS_DIR}/val/wider_val.txt")


class FaceDataset(Dataset):
    def __init__(self, img_list, image_size=640):
        self.image_size = image_size
        self.img_dir = os.path.dirname(img_list)
        with open(img_list, "r") as fr:
            self.imgs_path = fr.read().split()

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, idx):
        rel_path = self.imgs_path[idx].lstrip("/")
        img_path = os.path.join(self.img_dir, "images", rel_path)
        im = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if im is None:
            raise FileNotFoundError(img_path)
        h0, w0 = im.shape[:2]
        img = letterbox_image(im, self.image_size)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        return img, rel_path, (h0, w0)

    @staticmethod
    def collate_fn(batch):
        im, path, orig_shapes = zip(*batch)
        return np.concatenate([i[None] for i in im], axis=0), path, orig_shapes


def postprocess(pred, orig_shapes, conf_thres=0.01, iou_thres=0.5, max_det=2048, imgsz=640):
    """Ultralytics ops.non_max_suppression + ops.scale_boxes (same as YOLO.predict)."""
    if not torch.is_tensor(pred):
        pred = torch.from_numpy(np.ascontiguousarray(pred))
    if pred.ndim != 3:
        raise ValueError(f"unexpected output rank {pred.ndim}, shape={tuple(pred.shape)}")
    if pred.shape[1] > pred.shape[2]:
        pred = pred.transpose(1, 2)

    preds = ops.non_max_suppression(
        pred,
        conf_thres,
        iou_thres,
        nc=1,
        agnostic=True,
        max_det=max_det,
    )

    results = []
    img1_shape = (imgsz, imgsz)
    for det, orig_hw in zip(preds, orig_shapes):
        if det.numel():
            det = det.clone()
            det[:, :4] = ops.scale_boxes(img1_shape, det[:, :4], orig_hw)
            xyxy_conf = det[:, :5].detach().cpu().numpy()
        else:
            xyxy_conf = np.zeros((0, 5), dtype=np.float32)
        results.append(xyxy_conf)
    return results


def save_widerface_txt(dets, img_name, save_folder):
    save_name = os.path.join(save_folder, img_name[:-4] + ".txt")
    dirname = os.path.dirname(save_name)
    os.makedirs(dirname, exist_ok=True)
    file_stem = os.path.basename(save_name)[:-4]
    with open(save_name, "w") as fd:
        fd.write(file_stem + "\n")
        fd.write(str(dets.shape[0]) + "\n")
        for box in dets:
            x1, y1, x2, y2, conf = box
            x1 = int(x1 + 0.5)
            y1 = int(y1 + 0.5)
            w = int(x2 - x1 + 0.5)
            h = int(y2 - y1 + 0.5)
            fd.write("%d %d %d %d %.03f\n" % (x1, y1, w, h, min(float(conf), 1.0)))


def get_dataloader(args):
    data_path = resolve_val_list(args.datasets)
    datasets = FaceDataset(data_path, args.imgsz)
    return torch.utils.data.DataLoader(
        datasets,
        batch_size=args.batchsize,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=datasets.collate_fn,
    )


def main():
    args = parse_args()

    batch_size = args.batchsize

    target = tvm.target.iluvatar(model="MR", options="-libs=cudnn,cublas,ixinfer")
    device = tvm.device(target.kind.name, 0)

    lib = tvm.runtime.load_module(args.engine)
    module = tvm.contrib.graph_executor.GraphModule(lib["default"](device))

    if args.perf_only:
        ftimer = module.module.time_evaluator("run", device, number=100, repeat=1)
        prof_res = np.array(ftimer().results) * 1000
        fps = batch_size * 1000 / np.mean(prof_res)
        print(f"\n* Mean inference time: {np.mean(prof_res):.3f} ms, Mean fps: {fps:.3f}")
        return

    for _ in range(args.warmup):
        module.run()

    dataloader = get_dataloader(args)
    save_folder = "./widerface_txt/"

    for batch in tqdm(dataloader):
        image, img_names, orig_shapes = batch

        pad_batch = len(image) != batch_size
        if pad_batch:
            origin_size = len(image)
            image = np.resize(image, (batch_size, *image.shape[1:]))

        module.set_input(args.input_name, tvm.nd.array(image, device))
        module.run()
        outputs = module.get_output(0).asnumpy()

        if pad_batch:
            outputs = outputs[:origin_size]

        dets_bs = postprocess(
            outputs, orig_shapes, args.conf, args.iou, args.max_det, args.imgsz)
        for dets, img_name in zip(dets_bs, img_names):
            save_widerface_txt(dets, img_name, save_folder)


if __name__ == "__main__":
    main()
