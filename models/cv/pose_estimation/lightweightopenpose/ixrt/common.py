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
import glob
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
from modules.keypoints import extract_keypoints, group_keypoints
from torch.utils.data.dataset import Dataset
import json


def check_target(inference, target):
    satisfied = False
    if inference > target:
        satisfied = True  
    return satisfied

def preprocess_img(img_path, img_sz):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = F.resize(img, 256, Image.BILINEAR)
    img = F.center_crop(img, img_sz)
    img = F.to_tensor(img)
    img = F.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)
    img = img.permute(1, 2, 0)
    img = np.asarray(img, dtype='float32')
    return img

def get_dataloader(datasets_dir, bsz, imgsz, label_file_name="val_map.txt"):
    label_file = os.path.join(datasets_dir, label_file_name)
    with open(label_file, "r") as f:
        label_data = f.readlines()
    label_dict = dict()
    for line in label_data:
        line = line.strip().split('\t')
        label_dict[line[0]] = int(line[1])

    files = os.listdir(datasets_dir)
    batch_img, batch_label = [], []

    for file in files:
        if file == label_file_name:
            continue
        file_path = os.path.join(datasets_dir, file)
        img = preprocess_img(file_path, imgsz)
        batch_img.append(np.expand_dims(img, 0))
        batch_label.append(label_dict[file])
        if len(batch_img) == bsz:
            yield np.concatenate(batch_img, 0), np.array(batch_label)
            batch_img, batch_label = [], []

    if len(batch_img) > 0:
        yield np.concatenate(batch_img, 0), np.array(batch_label)

def eval_batch(batch_score, batch_label):
    batch_score = torch.from_numpy(batch_score)
    values, indices = batch_score.topk(5)
    top1, top5 = 0, 0
    for idx, label in enumerate(batch_label):

        if label == indices[idx][0]:
            top1 += 1
        if label in indices[idx]:
            top5 += 1
    return top1, top5



def run_coco_eval(gt_file_path, dt_file_path):
    annotation_type = 'keypoints'
    print('Running test for {} results.'.format(annotation_type))

    coco_gt = COCO(gt_file_path)
    coco_dt = coco_gt.loadRes(dt_file_path)

    result = COCOeval(coco_gt, coco_dt, annotation_type)
    result.evaluate()
    result.accumulate()
    result.summarize()


def normalize(img, img_mean, img_scale):
    img = np.array(img, dtype=np.float32)
    img = (img - img_mean) * img_scale
    return img


def pad_width(img, stride, pad_value, min_dims):
    h, w, _ = img.shape
    h = min(min_dims[0], h)
    min_dims[0] = math.ceil(min_dims[0] / float(stride)) * stride
    min_dims[1] = max(min_dims[1], w)
    min_dims[1] = math.ceil(min_dims[1] / float(stride)) * stride
    pad = []
    pad.append(int(math.floor((min_dims[0] - h) / 2.0)))
    pad.append(int(math.floor((min_dims[1] - w) / 2.0)))
    pad.append(int(min_dims[0] - h - pad[0]))
    pad.append(int(min_dims[1] - w - pad[1]))
    padded_img = cv2.copyMakeBorder(img, pad[0], pad[2], pad[1], pad[3],
                                    cv2.BORDER_CONSTANT, value=pad_value)
    return padded_img, pad


def convert_to_coco_format(pose_entries, all_keypoints):
    coco_keypoints = []
    scores = []
    for n in range(len(pose_entries)):
        if len(pose_entries[n]) == 0:
            continue
        keypoints = [0] * 17 * 3
        to_coco_map = [0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
        person_score = pose_entries[n][-2]
        position_id = -1
        for keypoint_id in pose_entries[n][:-2]:
            position_id += 1
            if position_id == 1:  # no 'neck' in COCO
                continue

            cx, cy, score, visibility = 0, 0, 0, 0  # keypoint not found
            if keypoint_id != -1:
                cx, cy, score = all_keypoints[int(keypoint_id), 0:3]
                cx = cx + 0.5
                cy = cy + 0.5
                visibility = 1
            keypoints[to_coco_map[position_id] * 3 + 0] = cx
            keypoints[to_coco_map[position_id] * 3 + 1] = cy
            keypoints[to_coco_map[position_id] * 3 + 2] = visibility
        coco_keypoints.append(keypoints)
        scores.append(person_score * max(0, (pose_entries[n][-1] - 1)))  # -1 for 'neck'
    return coco_keypoints, scores


class CocoValDataset(Dataset):
    def __init__(self, labels, images_folder):
        super().__init__()
        with open(labels, 'r') as f:
            self._labels = json.load(f)
        self._images_folder = images_folder

    def __getitem__(self, idx):
        file_name = self._labels['images'][idx]['file_name']
        img = cv2.imread(os.path.join(self._images_folder, file_name), cv2.IMREAD_COLOR)
        return {
            'img': img,
            'file_name': file_name
        }

    def __len__(self):
        return len(self._labels['images'])

