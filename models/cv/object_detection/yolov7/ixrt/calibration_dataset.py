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
import torch
import torchvision.datasets
from torch.utils.data import DataLoader
from datasets.coco import CocoDetection

def create_dataloaders(data_path, annFile, img_sz=640, batch_size=32, step=32, workers=2, data_process_type="yolov5"):
    dataset = CocoDetection(
        root=data_path,
        annFile=annFile,
        img_size=img_sz,
        data_process_type=data_process_type
    )
    calibration_dataset = dataset
    num_samples = min(5000, batch_size * step)
    if num_samples > 0:
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
    return calibration_dataloader