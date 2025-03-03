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
from torch.utils.data import DataLoader, Dataset
from typing import Any, Callable, List, Optional, Tuple
import cv2
import numpy as np
import pycocotools
from img_preprocess import preprocess_img
from tqdm import tqdm

class CocoDataset(Dataset):
    """
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
    """
    def __init__(
        self,
        root: str,
        annFile: str,
        img_size: tuple,
    ) -> None:
        super().__init__()
        from pycocotools.coco import COCO

        self.root_dir = root
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.img_size = img_size

        self.transforms = preprocess_img

    def _load_image(self, id: int):
        path = self.coco.loadImgs(id)[0]["file_name"]
        img_path = os.path.join(self.root_dir, path)
        data = cv2.imread(img_path)
        return data, img_path

    def _load_target(self, id: int) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image, path = self._load_image(id)
        target = self._load_target(id)
        origin_shape = image.shape[:2]
        # print("read image shape :", image.shape)

        if self.transforms is not None:
            image, paddings = self.transforms(image, self.img_size[1], self.img_size[0])

        if len(target) > 0:
            image_id = target[0]["image_id"]
        else:
            # have no target
            image_id = -1
        return image, torch.tensor(origin_shape), image_id, torch.tensor(paddings), path

    def __len__(self) -> int:
        return len(self.ids)


def create_dataloaders(data_path, annFile, img_h=800, img_w=1067, batch_size=1, workers=1):
    dataset = CocoDataset(
        root=data_path,
        annFile=annFile,
        img_size=(img_h, img_w)
    )
    eval_dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=batch_size,
        drop_last=False,
        num_workers=workers,
    )
    return eval_dataloader

labels = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
    return [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
        35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
        64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

