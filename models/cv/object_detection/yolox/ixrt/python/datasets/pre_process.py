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
import cv2
import math
import numpy as np

from .common import letterbox

def get_post_process(data_process_type):
    if data_process_type == "yolov5":
        return Yolov5Preprocess
    elif data_process_type == "yolov3":
        return Yolov3Preprocess
    elif data_process_type == "yolox":
        return YoloxPreprocess
    return None

def Yolov3Preprocess(image, img_size):

    h0, w0 = image.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # ratio

    image = cv2.resize(image, (img_size, img_size))
    image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    image = np.ascontiguousarray(image).astype(np.float32) / 255.0  # 0~1 np array
    return image

def Yolov5Preprocess(image, img_size, augment=False):

    h0, w0 = image.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # ratio

    if r != 1:  # if sizes are not equal
        interp = cv2.INTER_LINEAR if (augment or r > 1) else cv2.INTER_AREA
        image = cv2.resize(image, (math.ceil(w0 * r), math.ceil(h0 * r)), interpolation=interp)

    # shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  rect == True

    image, ratio, dwdh = letterbox(image, new_shape=img_size, auto=False, scaleup=False)
    image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    image = np.ascontiguousarray(image).astype(np.float32) / 255.0  # 0~1 np array
    return image

def YoloxPreprocess(img, img_size, swap=(2,0,1)):

    padded_img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 114
    r = min(img_size / img.shape[0], img_size / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR, 
    ).astype(np.uint8)

    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)

    return padded_img