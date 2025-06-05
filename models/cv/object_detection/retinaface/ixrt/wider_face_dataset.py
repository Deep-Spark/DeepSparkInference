# Copyright (c) 2025, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
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
import torch
import torch.utils.data as data
import cv2
import numpy as np

def lt_preproc(img, input_size=(320, 320), swap=(2, 0, 1), mean=(104.0, 117.0, 123.0), std=1.0):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.float32) * 114.0
    else:
        padded_img = np.ones(input_size, dtype=np.float32) * 114.0

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR, 
    )

    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img -= mean
    padded_img /= std

    if swap is not None:
        padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)

    return padded_img, r


class WiderFaceDetection(data.Dataset):
    def __init__(self, prj_dir, preproc=lt_preproc, input_size=(320, 320)):
        self.preproc = preproc
        self.input_size = input_size
        self.image_dir = os.path.join(prj_dir, "val/images")

        testset_list = os.path.join(prj_dir, "val/wider_val.txt")
        with open(testset_list, 'r') as fr:
            self.imgs_path = fr.read().split()

        self.count = 0

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        img_file = os.path.join(self.image_dir, self.imgs_path[index])
        img = cv2.imread(img_file, cv2.IMREAD_COLOR)

        if self.preproc is not None:
            img, r = self.preproc(img, self.input_size)

        return torch.from_numpy(img), r, self.imgs_path[index]


def detection_collate(batch):
    imgs, rs, img_files = list(), list(), list()
    for i, (img, scale, img_file) in enumerate(batch):
        if torch.is_tensor(img):
            imgs.append(img)
        rs.append(scale)
        img_files.append(img_file)

    imgs = torch.stack(imgs, 0)
    return imgs, torch.from_numpy(np.array(rs)), img_files