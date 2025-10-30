# Copyright, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
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
import cv2
import json
import torch
import torchvision
import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

coco80_to_coco91 = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
    23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
    65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88,
    89, 90
]

coco80_to_coco91_dict = {idx: i for idx, i in enumerate(coco80_to_coco91)}
coco91_to_coco80_dict = {i: idx for idx, i in enumerate(coco80_to_coco91)}

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    # Resize and pad image while meeting stride-multiple constraints
    # current shape [height, width]
    
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im,
                            top,
                            bottom,
                            left,
                            right,
                            cv2.BORDER_CONSTANT,
                            value=color)
    return im, ratio, (dw, dh)

def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y

def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    if clip:
        clip_boxes(x, (h - eps, w - eps))  # warning: inplace clip
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def box_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])

def box_iou(box1, box2, eps=1e-7):
    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1[:, None].chunk(2, 2), box2.chunk(2, 1)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / (box_area(box1.T)[:, None] + box_area(box2.T) - inter + eps)

def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0],
                   img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
            img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[:, [0, 2]] -= pad[0]  # x padding
    boxes[:, [1, 3]] -= pad[1]  # y padding
    boxes[:, :4] /= gain
    clip_boxes(boxes, img0_shape)

    return boxes


class COCO2017Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 image_dir_path,
                 label_json_path,
                 image_size=640,
                 pad_color=114,
                 val_mode=True,
                 input_layout="NCHW"):

        self.image_dir_path = image_dir_path
        self.label_json_path = label_json_path
        self.image_size = image_size
        self.pad_color = pad_color
        self.val_mode = val_mode
        self.input_layout = input_layout

        self.coco = COCO(annotation_file=self.label_json_path)
        
        if self.val_mode:
            self.img_ids = list(sorted(self.coco.imgs.keys()))
        else:
            self.img_ids = sorted(list(self.coco.imgToAnns.keys()))

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        img_path = self._get_image_path(index)
        img, (h0, w0), (h, w) = self._load_image(index)

        img, ratio, pad = letterbox(img,
                                    self.image_size,
                                    color=(self.pad_color, self.pad_color, self.pad_color))
        shapes = (h0, w0), ((h / h0, w / w0), pad)

        # load label
        raw_label = self._load_json_label(index)
        # normalized xywh to pixel xyxy format
        raw_label[:, 1:] = xywhn2xyxy(raw_label[:, 1:],
                                      ratio[0] * w,
                                      ratio[1] * h,
                                      padw=pad[0],
                                      padh=pad[1])

        raw_label[:, 1:] = xyxy2xywhn(raw_label[:, 1:],
                                      w=img.shape[1],
                                      h=img.shape[0],
                                      clip=True,
                                      eps=1E-3)

        nl = len(raw_label)
        labels_out = np.zeros((nl, 6))
        labels_out[:, 1:] = raw_label

        # HWC to CHW, BGR to RGB
        img = img.transpose((2, 0, 1))[::-1] 
        img = np.ascontiguousarray(img) / 255.0
        if self.input_layout == "NHWC":
            img = img.transpose((1, 2, 0))

        return img, labels_out, img_path, shapes

    def _get_image_path(self, index):
        idx = self.img_ids[index]
        path = self.coco.loadImgs(idx)[0]["file_name"]
        img_path = os.path.join(self.image_dir_path, path)
        return img_path

    def _load_image(self, index):
        img_path = self._get_image_path(index)

        im = cv2.imread(img_path)
        h0, w0 = im.shape[:2]
        r = self.image_size / max(h0, w0)
        if r != 1:
            im = cv2.resize(im, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_LINEAR)
        return im.astype("float32"), (h0, w0), im.shape[:2]
    
    def _load_json_label(self, index):
        _, (h0, w0), _ = self._load_image(index)

        idx = self.img_ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=idx)
        targets = self.coco.loadAnns(ids=ann_ids)

        labels = []
        for target in targets:
            cat = target["category_id"]
            coco80_cat = coco91_to_coco80_dict[cat]
            cat = np.array([[coco80_cat]])

            x, y, w, h = target["bbox"]
            x1, y1, x2, y2 = x, y, int(x + w), int(y + h)
            xyxy = np.array([[x1, y1, x2, y2]])
            xywhn = xyxy2xywhn(xyxy, w0, h0)
            labels.append(np.hstack((cat, xywhn)))

        if labels:
            labels = np.vstack(labels)
        else:
            if self.val_mode:
                labels = np.zeros((1, 5))
            else:
                raise ValueError(f"set val_mode = False to use images with labels")

        return labels

    @staticmethod
    def collate_fn(batch):
        im, label, path, shapes = zip(*batch)
        for i, lb in enumerate(label):
            lb[:, 0] = i
        return np.concatenate([i[None] for i in im], axis=0), np.concatenate(label, 0), path, shapes


def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=True,
        labels=(),
        max_det=300,
        nm=0,  # number of masks
):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    if isinstance(prediction, (list, tuple)):  
        prediction = prediction[0]

    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    mi = 5 + nc 
    output = [torch.zeros((0, 6 + nm))] * bs
    for xi, x in enumerate(prediction):
  
        x = x[xc[xi]]

        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]
            v[:, 4] = 1.0
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]

        # Box/Mask
        box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, mi:]

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat(
                (box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask),
                          1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]
        else:
            x = x[x[:, 4].argsort(descending=True)]
        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:
            i = i[:max_det]
        if merge and (1 < n < 3E3):
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres
            weights = iou * scores[None]
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)
            if redundant:
                i = i[iou.sum(1) > 1]

        output[xi] = x[i]
    return output

def get_coco_accuracy(pred_json, ann_json):
    coco = COCO(annotation_file=ann_json)
    coco_pred = coco.loadRes(pred_json)

    coco_evaluator = COCOeval(cocoGt=coco, cocoDt=coco_pred, iouType="bbox")
            
    coco_evaluator.evaluate()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    return coco_evaluator.stats

class COCO2017Evaluator:    
    def __init__(self,
                 label_path,
                 image_size=640,
                 conf_thres=0.001,
                 iou_thres=0.65):
        
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.label_path = label_path
        self.image_size = image_size

        self.jdict = []

        # iou vector for mAP@0.5:0.95
        self.iouv = torch.linspace(0.5, 0.95, 10)  
        self.niou = self.iouv.numel()
    
    def evaluate(self, pred, all_inputs, nms_count=None):
        im = all_inputs[0]
        targets = all_inputs[1]
        paths = all_inputs[2]
        shapes = all_inputs[3]

        _, _, height, width = im.shape
        targets[:, 2:] *= np.array((width, height, width, height))
        
        pred = torch.from_numpy(pred)
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)

        for idx, det in enumerate(pred):
            img_path = paths[idx]

            predn = det
            shape = shapes[idx][0]
            scale_boxes(im[idx].shape[1:], predn[:, :4], shape, shapes[idx][1])  # native-space pred
            self._save_one_json(predn, self.jdict, img_path, coco80_to_coco91)  # append to COCO-JSON dictionary
        

    def _save_one_json(self, predn, jdict, path, class_map):
        # Save one JSON result in the format
        image_id = int(os.path.splitext(os.path.basename(path))[0])
        box = xyxy2xywh(predn[:, :4])
        box[:, :2] -= box[:, 2:] / 2
        for p, b in zip(predn.tolist(), box.tolist()):
            jdict.append({
                'image_id': image_id,
                'category_id': class_map[int(p[5])],
                'bbox': [round(x, 3) for x in b],
                'score': round(p[4], 5)
            })


    def summary(self):
        if len(self.jdict):
            pred_json = os.path.join("coco2017_predictions.json")
            with open(pred_json, 'w') as f:
                json.dump(self.jdict, f)
            result = get_coco_accuracy(pred_json, self.label_path)
        else:
            raise ValueError("can not find generated json dict for pycocotools")
        return result

