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

#!/usr/bin/env python
# -*- coding: utf-8 -*-
from copy import deepcopy

import cv2
import numpy as np
import pycocotools.mask as mask_util

DETECTIONS_PER_IMAGE = 100
SCORE_THRESH = 0.05

def postprocess_img(
    w_ori,
    h_ori,
    input_w,
    input_h,
    scores_h,
    boxes_h,
    classes_h,
    masks_h,
    pad_list
):
    # h_ori, w_ori, _ = img.shape
    h_ratio = (h_ori) / (input_h - (pad_list[2] + pad_list[3]))
    w_ratio = (w_ori) / (input_w - (pad_list[0] + pad_list[1]))
    bboxs_masks = []
    for i in range(DETECTIONS_PER_IMAGE):
        if scores_h[i, 0] > SCORE_THRESH:
            x1 = int((boxes_h[i, 0] - pad_list[0]) * w_ratio)
            y1 = int((boxes_h[i, 1] - pad_list[2]) * h_ratio)
            x2 = int((boxes_h[i, 2] - pad_list[0]) * w_ratio)
            y2 = int((boxes_h[i, 3] - pad_list[2]) * h_ratio)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w_ori, x2)
            y2 = min(h_ori, y2)
            if x1 >= x2 or y1 >=y2:
                continue

            label = int(classes_h[i, 0])
            score = scores_h[i, 0]
            
            maskPart = deepcopy(masks_h[i, 0])
            maskPart = (maskPart * 255.0).astype(np.uint8)
            # print("xyxy : ", x1, y1, x2 , y2 )
            maskPart = cv2.resize(maskPart, (x2 - x1, y2 - y1))
            curMask = np.ones((h_ori, w_ori), dtype=np.uint8)
            _, maskPart = cv2.threshold(maskPart, 127, 255, cv2.THRESH_BINARY)
            curMask[y1:y2, x1:x2] = maskPart
            curMask = curMask.astype(np.uint8)
            _, curMask = cv2.threshold(curMask, 127, 255, cv2.THRESH_BINARY)
            
            rle = mask_util.encode(np.array(curMask[:, :, np.newaxis], dtype=np.uint8, order="F"))[0]
            rle['counts'] = rle["counts"].decode("utf-8")
            bboxs_masks.append([ x1, y1, x2-x1, y2-y1, label, score, rle])
    return bboxs_masks

def postprocess_img_and_save(
    img,
    input_w,
    input_h,
    scores_h,
    boxes_h,
    classes_h,
    masks_h,
    pad_list,
    save_img_name,
):
    h_ori, w_ori, _ = img.shape
    h_ratio = (h_ori) / (input_h - (pad_list[2] + pad_list[3]))
    w_ratio = (w_ori) / (input_w - (pad_list[0] + pad_list[1]))

    for i in range(DETECTIONS_PER_IMAGE):
        if scores_h[i, 0] > SCORE_THRESH:
            x1 = int((boxes_h[i, 0] - pad_list[0]) * w_ratio)
            y1 = int((boxes_h[i, 1] - pad_list[2]) * h_ratio)
            x2 = int((boxes_h[i, 2] - pad_list[0]) * w_ratio)
            y2 = int((boxes_h[i, 3] - pad_list[2]) * h_ratio)
            label = int(classes_h[i, 0])
            score = scores_h[i, 0]
            print(
                "boxes:{:4},{:4},{:4},{:4}\tscores:{:.4f}\tlabel:{:3}".format(
                    x1, y1, x2, y2, score, label
                )
            )
            cv2.rectangle(img, (x1, y1, x2 - x1, y2 - y1), (0x27, 0xC1, 0x36), 2)
            cv2.putText(
                img,
                str(label),
                (x1, y1 - 1),
                cv2.FONT_HERSHEY_PLAIN,
                1.2,
                (0xFF, 0xFF, 0xFF),
                2,
            )
            maskPart = deepcopy(masks_h[i, 0])
            maskPart = (maskPart * 255.0).astype(np.uint8)
            maskPart = cv2.resize(maskPart, (x2 - x1, y2 - y1))
            curMask = np.ones((h_ori, w_ori), dtype=np.uint8)
            _, maskPart = cv2.threshold(maskPart, 127, 255, cv2.THRESH_BINARY)
            curMask[y1:y2, x1:x2] = maskPart
            curMask = curMask.astype(np.uint8)
            _, curMask = cv2.threshold(curMask, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(
                curMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )

            img = cv2.drawContours(img, contours[0], -1, (0, 0, 255), 2)

    cv2.imwrite(save_img_name, img)

    return img
