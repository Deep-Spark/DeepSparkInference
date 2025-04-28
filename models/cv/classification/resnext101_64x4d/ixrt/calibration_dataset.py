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
import torchvision.datasets
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import transforms as T


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
        if self.transform is not None:
            sample = self.transform(sample)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        img_name = os.path.basename(path)
        target = self.img2label_map[img_name]

        return sample, target


def create_dataloaders(data_path, num_samples=1024, img_sz=224, batch_size=2, workers=0):
    dataset = CalibrationImageNet(
        data_path,
        transform=T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(img_sz),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
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


def getdataloader(dataset_dir, step=20, batch_size=32, workers=2, img_sz=224, total_sample=50000):
    num_samples = min(total_sample, step * batch_size)
    if step < 0:
        num_samples = None
    calibration_dataloader, _ = create_dataloaders(
        dataset_dir,
        img_sz=img_sz,
        batch_size=batch_size,
        workers=workers,
        num_samples=num_samples,
    )
    return calibration_dataloader