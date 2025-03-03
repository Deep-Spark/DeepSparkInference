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
import math
from PIL import Image

import numpy as np
import torch
import torchvision.datasets
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import transforms as T

class ToSpaceBGR(object):
    def __init__(self, is_bgr):
        self.is_bgr = is_bgr

    def __call__(self, tensor):
        if self.is_bgr:
            new_tensor = tensor.clone()
            new_tensor[0] = tensor[2]
            new_tensor[2] = tensor[0]
            tensor = new_tensor
        return tensor

class ToRange255(object):
    def __init__(self, is_255):
        self.is_255 = is_255

    def __call__(self, tensor):
        if self.is_255:
            tensor.mul_(255)
        return tensor

class TransformImage(object):
    def __init__(self, imgsize=300, scale=0.875,
                 preserve_aspect_ratio=True):
        self.input_size = [3, imgsize, imgsize]
        self.input_space = 'RGB'
        self.input_range = [0, 1]
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]

        # https://github.com/tensorflow/models/blob/master/research/inception/inception/image_processing.py#L294
        self.scale = scale
        
        tfs = []
        if preserve_aspect_ratio:
            tfs.append(T.Resize(int(math.floor(max(self.input_size)/self.scale))))
        else:
            height = int(self.input_size[1] / self.scale)
            width = int(self.input_size[2] / self.scale)
            tfs.append(T.Resize((height, width)))

        
        tfs.append(T.CenterCrop(max(self.input_size)))
        tfs.append(T.ToTensor())
        tfs.append(ToSpaceBGR(self.input_space=='BGR'))
        tfs.append(ToRange255(max(self.input_range)==255))
        tfs.append(T.Normalize(mean=self.mean, std=self.std))

        self.tf = T.Compose(tfs)

    def __call__(self, img):
        tensor = self.tf(img)
        return tensor

class CalibrationImageNet(torchvision.datasets.ImageFolder):
    def __init__(self, *args, **kwargs):
        super(CalibrationImageNet, self).__init__(*args, **kwargs)
        img2label_path = os.path.join(self.root, "val_map.txt")
        if not os.path.exists(img2label_path):
            raise FileNotFoundError(f"Not found label file `{img2label_path}`.")

        self.img2label_map = self.make_img2label_map(img2label_path)

    def make_img2label_map(self, path):
        with open(path) as f:
            lines = f.readlines()

        img2lable_map = dict()
        for line in lines:
            line = line.lstrip().rstrip().split("\t")
            if len(line) != 2:
                continue
            img_name, label = line
            img_name = img_name.strip()
            if img_name in [None, ""]:
                continue
            label = int(label.strip())
            img2lable_map[img_name] = label
        return img2lable_map

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        sample = sample.convert("RGB")
        if self.transform is not None:
            sample = self.transform(sample)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        img_name = os.path.basename(path)
        target = self.img2label_map[img_name]

        return sample, target


def create_dataloaders(data_path, transforms, num_samples=1024, img_sz=300, batch_size=64, workers=0):
    dataset = CalibrationImageNet(
        data_path,
        transform=transforms,
    )

    calibration_dataset = dataset
    if num_samples is not None:
        calibration_dataset = torch.utils.data.Subset(
            dataset, indices=range(num_samples)
        )

    calibration_dataloader = DataLoader(
        calibration_dataset,
        shuffle=False,
        batch_size=batch_size,
        drop_last=False,
        num_workers=workers,
    )

    verify_dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=batch_size,
        drop_last=False,
        num_workers=workers,
    )

    return calibration_dataloader, verify_dataloader


def getdataloader(dataset_dir, transforms, step=20, batch_size=64, workers=0, img_sz=299, total_sample=50000):
    num_samples = min(total_sample, step * batch_size)
    if step < 0:
        num_samples = None
    calibration_dataloader, _ = create_dataloaders(
        dataset_dir,
        transforms,
        img_sz=img_sz,
        batch_size=batch_size,
        workers=workers,
        num_samples=num_samples,
    )
    return calibration_dataloader