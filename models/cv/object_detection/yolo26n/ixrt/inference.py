# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os

import cuda.cudart as cudart
import cv2
import numpy as np
import ixrt


class Engine:
    def __init__(self, path):
        self.logger = ixrt.Logger(ixrt.Logger.ERROR)
        self.runtime = ixrt.Runtime(self.logger)
        with open(path, "rb") as handle:
            self.engine = self.runtime.deserialize_cuda_engine(handle.read())
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.allocations = [], [], []
        for index in range(self.engine.num_bindings):
            dtype = np.dtype(ixrt.nptype(self.engine.get_binding_dtype(index)))
            shape = list(self.context.get_binding_shape(index))
            nbytes = int(dtype.itemsize * np.prod(shape))
            err, allocation = cudart.cudaMalloc(nbytes)
            assert err == cudart.cudaError_t.cudaSuccess
            binding = {"dtype": dtype, "shape": shape, "allocation": allocation, "nbytes": nbytes}
            self.allocations.append(allocation)
            target = self.inputs if self.engine.binding_is_input(index) else self.outputs
            target.append(binding)

    def shape(self):
        return [int(value) for value in self.inputs[0]["shape"]]

    def run(self, data):
        data = np.ascontiguousarray(data.astype(self.inputs[0]["dtype"]))
        (err,) = cudart.cudaMemcpy(
            self.inputs[0]["allocation"], data, self.inputs[0]["nbytes"],
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
        )
        assert err == cudart.cudaError_t.cudaSuccess
        self.context.execute_v2(self.allocations)
        out = self.outputs[0]
        result = np.zeros(out["shape"], out["dtype"])
        (err,) = cudart.cudaMemcpy(
            result, out["allocation"], out["nbytes"],
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        )
        assert err == cudart.cudaError_t.cudaSuccess
        return result

    def close(self):
        for binding in self.inputs + self.outputs:
            cudart.cudaFree(binding["allocation"])

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

parser = argparse.ArgumentParser(description="YOLO26n COCO val2017 accuracy with ixRT")
parser.add_argument("--engine", required=True, help="path to the built ixRT engine.")
parser.add_argument("--datasets", required=True, help="path to the COCO directory.")
parser.add_argument("--limit", type=int, default=0,
                    help="evaluate only the first N images; 0 means all of them.")
args = parser.parse_args()

engine_path = args.engine
dataset = args.datasets
limit = args.limit
CONF, IOU, MAX_DET, MAX_NMS, MAX_WH = 0.001, 0.7, 300, 30000, 7680


def first(*paths):
    return next((path for path in paths if os.path.exists(path)), "")


annotation = first(os.path.join(dataset, "annotations", "instances_val2017.json"))
images_dir = first(os.path.join(dataset, "val2017"), os.path.join(dataset, "images", "val2017"))
if not annotation or not images_dir:
    raise SystemExit("COCO 目录里需要 annotations/instances_val2017.json 和 val2017/（或 images/val2017/）")


def letterbox(raw, size):
    height, width = raw.shape[:2]
    ratio = min(size / height, size / width)
    new_w, new_h = round(width * ratio), round(height * ratio)
    pad_w, pad_h = (size - new_w) / 2, (size - new_h) / 2
    if (new_w, new_h) != (width, height):
        raw = cv2.resize(raw, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = round(pad_h - 0.1), round(pad_h + 0.1)
    left, right = round(pad_w - 0.1), round(pad_w + 0.1)
    raw = cv2.copyMakeBorder(raw, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    tensor = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB).astype("float32").transpose(2, 0, 1) / 255.0
    return tensor, ratio, left, top


def nms(boxes, scores):
    order = scores.argsort()[::-1]
    keep = []
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    while order.size and len(keep) < MAX_DET:
        index = order[0]
        keep.append(index)
        rest = order[1:]
        x1 = np.maximum(boxes[index, 0], boxes[rest, 0])
        y1 = np.maximum(boxes[index, 1], boxes[rest, 1])
        x2 = np.minimum(boxes[index, 2], boxes[rest, 2])
        y2 = np.minimum(boxes[index, 3], boxes[rest, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        iou = inter / (areas[index] + areas[rest] - inter + 1e-9)
        order = rest[iou <= IOU]
    return np.asarray(keep, dtype=int)


def detections(raw):
    """(boxes xyxy, scores, classes) in letterboxed pixels, at most MAX_DET of them."""
    data = np.asarray(raw, dtype="float32")
    if data.ndim != 2:
        raise SystemExit("不认识的检测输出形状 " + str(list(data.shape)))
    if data.shape[1] == 6 and data.shape[0] > 6:
        # End-to-end export (NMS-free heads): x1, y1, x2, y2, score, class.
        data = data[data[:, 4] > CONF]
        return data[:, :4], data[:, 4], data[:, 5].astype(int)
    if data.shape[0] < data.shape[1]:
        data = data.T
    if data.shape[1] != 84:
        raise SystemExit("检测输出不是 4 + 80 类的 YOLO 布局：" + str(list(data.shape)))
    xy, wh, scores = data[:, :2], data[:, 2:4], data[:, 4:]
    boxes = np.concatenate([xy - wh / 2, xy + wh / 2], axis=1)
    rows, classes = np.nonzero(scores > CONF)
    values = scores[rows, classes]
    if values.size > MAX_NMS:
        top = values.argsort()[::-1][:MAX_NMS]
        rows, classes, values = rows[top], classes[top], values[top]
    picked = boxes[rows]
    keep = nms(picked + classes[:, None] * MAX_WH, values)
    return picked[keep], values[keep], classes[keep]


coco = COCO(annotation)
category_ids = sorted(coco.getCatIds())
image_ids = sorted(coco.getImgIds())
if limit > 0:
    image_ids = image_ids[:limit]

runner = Engine(engine_path)
count, _, size, _ = runner.shape()
found = []
evaluated = []
for position in range(0, len(image_ids), count):
    chunk = image_ids[position:position + count]
    data = np.zeros((count, 3, size, size), dtype="float32")
    meta = []
    for slot, image_id in enumerate(chunk):
        info = coco.loadImgs(image_id)[0]
        raw = cv2.imread(os.path.join(images_dir, info["file_name"]), cv2.IMREAD_COLOR)
        if raw is None:
            continue
        tensor, ratio, left, top = letterbox(raw, size)
        data[slot] = tensor
        meta.append((slot, image_id, ratio, left, top, raw.shape[1], raw.shape[0]))
    if not meta:
        continue
    output = np.asarray(runner.run(data))
    for slot, image_id, ratio, left, top, width, height in meta:
        evaluated.append(image_id)
        boxes, scores, classes = detections(output[slot])
        if not len(scores):
            continue
        boxes = boxes.copy()
        boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]] - left) / ratio, 0, width)
        boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]] - top) / ratio, 0, height)
        for box, score, cls in zip(boxes, scores, classes):
            found.append({
                "image_id": int(image_id),
                "category_id": category_ids[int(cls)],
                "bbox": [round(float(box[0]), 3), round(float(box[1]), 3),
                         round(float(box[2] - box[0]), 3), round(float(box[3] - box[1]), 3)],
                "score": round(float(score), 5),
            })
runner.close()

map50 = map50_95 = 0.0
if found:
    evaluator = COCOeval(coco, coco.loadRes(found), "bbox")
    evaluator.params.imgIds = evaluated
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    map50_95, map50 = float(evaluator.stats[0]), float(evaluator.stats[1])
print("=" * 56)
print(f"Images    : {len(evaluated)}")
print(f"Detections: {len(found)}")
print(f"IOU@0.5     : {map50:.4f}")
print(f"IOU@0.5:0.95: {map50_95:.4f}")
print("=" * 56)
if not evaluated or map50 <= 0:
    raise SystemExit("no detection was scored on COCO val2017")
