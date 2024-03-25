# Copyright (c) 2024, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
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

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

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
        img = self._load_image(img_path)

        img, r = self.preproc(img, input_size=self.image_size)
        
        return img, img_path, r

    def _get_image_path(self, index):
        idx = self.img_ids[index]
        path = self.coco.loadImgs(idx)[0]["file_name"]
        img_path = os.path.join(self.image_dir_path, path)
        return img_path

    def _load_image(self, img_path):
        img = cv2.imread(img_path)
        assert img is not None, f"file {img_path} not found"

        return img
    
    def preproc(self, img, input_size, swap=(2, 0, 1)):
        org_img = (img.shape[0], img.shape[1])
        img_ = cv2.resize(img, (input_size[0], input_size[1]))
        img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
        img_ = img_.transpose(swap) / 255.0
        img_ = np.ascontiguousarray(img_, dtype=np.float32)
        return img_, org_img

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
        im, img_path, r = zip(*batch)
        return np.concatenate([i[None] for i in im], axis=0), img_path, r

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
    
    def evaluate(self, pred, all_inputs):
        im = all_inputs[0]
        img_path = all_inputs[1]
        img_info = all_inputs[2]

        boxes = torch.squeeze(torch.from_numpy(pred[0]), dim=2)
        confs = torch.from_numpy(pred[1])
        detections = torch.cat((boxes, confs.float()), 2)

        nms_outputs = self.postprocess(
            detections, conf_thre=self.conf_thres, nms_thre=self.iou_thres
        )

        for (output, org_img, path) in zip(nms_outputs, img_info, img_path):
            if output is None:
                continue
            
            bboxes = output[:, 0:4]
            img_h, img_w = org_img
            bboxes[:, 0] *= img_w
            bboxes[:, 2] *= img_w
            bboxes[:, 1] *= img_h
            bboxes[:, 3] *= img_h

            cls = output[:, 5]
            scores = output[:, 4]
            
            bboxes = self._xyxy2xywh(bboxes)
            self._save_one_json(bboxes, cls, scores, self.jdict, path, coco80_to_coco91)
        
    def postprocess(self, prediction, num_classes=80, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
        output = [None for _ in range(len(prediction))]

        for i, image_pred in enumerate(prediction):
            # If none are remaining => process next image
            if not image_pred.size(0):
                continue
            # Get score and class with highest confidence
            class_conf, class_pred = torch.max(image_pred[:, 4: 4 + num_classes], 1, keepdim=True)

            conf_mask = (class_conf.squeeze() >= conf_thre).squeeze()
            detections = torch.cat((image_pred[:, :4], class_conf, class_pred.float()), 1)
            detections = detections[conf_mask]

            if not detections.size(0):
                continue
            if class_agnostic:
                nms_out_index = torchvision.ops.nms(
                    detections[:, :4],
                    detections[:, 4],
                    nms_thre,
                )
            else:
                nms_out_index = torchvision.ops.batched_nms(
                    detections[:, :4],
                    detections[:, 4],
                    detections[:, 5],
                    nms_thre,
                )
            detections = detections[nms_out_index]

            if output[i] is None:
                output[i] = detections
            else:
                output[i] = torch.cat((output[i], detections))

        return output
    
    def _xyxy2xywh(self, bboxes):
        bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
        bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
        return bboxes

    def _save_one_json(self, bboxes, class_, scores, jdict, path, class_map):
        image_id = int(os.path.splitext(os.path.basename(path))[0])
        for box, score, cls in zip(bboxes.numpy().tolist(), scores.numpy().tolist(), class_.numpy().tolist()):
            jdict.append({
                'image_id': image_id,
                'category_id': class_map[int(cls)],
                'bbox': box,
                'score': score
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
