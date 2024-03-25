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
import cv2
import numpy as np


def resize_img(img, input_w, input_h):
    rows, cols, _ = img.shape
    r_w = input_w / (cols * 1.0)
    r_h = input_h / (rows * 1.0)

    if r_h > r_w:
        w = int(input_w)
        h = int(r_w * rows)
        x = 0.0
        y = (input_h - h) / 2.0
    else:
        w = int(r_h * cols)
        h = int(input_h)
        x = (input_w - w) / 2.0
        y = 0.0
    X_LEFT_PAD = int((round(x - 0.1)))
    X_RIGHT_PAD = int((round(x + 0.1)))
    Y_TOP_PAD = int((round(y - 0.1)))
    Y_BOTTOM_PAD = int((round(y + 0.1)))

    re = np.zeros((h, w, 3), dtype=np.uint8)
    re = cv2.resize(img, (w, h), 0, 0, cv2.INTER_LINEAR)
    out = np.zeros((input_h, input_w, 3), dtype=np.uint8) + 128

    out[Y_TOP_PAD : (h + Y_TOP_PAD), X_LEFT_PAD : (w + X_LEFT_PAD), :] = re
    pad_list = [X_LEFT_PAD, X_RIGHT_PAD, Y_TOP_PAD, Y_BOTTOM_PAD]

    return out, pad_list


def preprocess_img(
    img, input_w, input_h, mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0]
):
    resized_img, pad_list = resize_img(img, input_w, input_h)
    mean = np.array(mean).reshape((1, 1, 3))
    std = np.array(std).reshape((1, 1, 3))

    resized_img = resized_img.astype(np.float32)
    resized_img = (resized_img - mean) / std
    resized_img = np.transpose(resized_img, (2, 0, 1)).astype(np.float32)
    return resized_img, pad_list
