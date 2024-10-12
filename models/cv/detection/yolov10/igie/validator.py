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
import tvm
import json
import torch
import numpy as np

from tqdm import tqdm

from ultralytics.models.yolov10 import YOLOv10DetectionValidator
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.metrics import ConfusionMatrix
from ultralytics.data.converter import coco80_to_coco91_class

class IGIE_Validator(YOLOv10DetectionValidator):
    def __call__(self, engine, device):
        self.data = check_det_dataset(self.args.data)
        self.dataloader = self.get_dataloader(self.data.get(self.args.split), self.args.batch)
        self.init_metrics()

        self.stats = {'tp': [], 'conf': [], 'pred_cls': [], 'target_cls': []}
        
        # wram up
        for _ in range(3):
            engine.run()

        for batch in tqdm(self.dataloader):
            batch = self.preprocess(batch)

            imgs = batch['img']
            pad_batch = len(imgs) != self.args.batch
            if pad_batch:
                origin_size = len(imgs)
                imgs = np.resize(imgs, (self.args.batch, *imgs.shape[1:]))
            
            engine.set_input(0, tvm.nd.array(imgs, device))
            
            engine.run()
            
            outputs = engine.get_output(0).asnumpy()

            if pad_batch:
                outputs = outputs[:origin_size]
            
            outputs = torch.from_numpy(outputs)
            
            preds = self.postprocess([outputs])
            
            self.update_metrics(preds, batch)
        
        stats = self.get_stats()

        if self.args.save_json and self.jdict:
            with open(str(self.save_dir / 'predictions.json'), 'w') as f:
                print(f'Saving {f.name} ...')
                json.dump(self.jdict, f)  # flatten and save

        stats = self.eval_json(stats)

        return stats

    def init_metrics(self):
        """Initialize evaluation metrics for YOLO."""
        val = self.data.get(self.args.split, '')  # validation path
        self.is_coco = isinstance(val, str) and 'coco' in val and val.endswith(f'{os.sep}val2017.txt')  # is COCO
        self.class_map = coco80_to_coco91_class() if self.is_coco else list(range(1000))
        self.args.save_json |= self.is_coco and not self.training  # run on final val if training COCO
        self.names = self.data['names']
        self.nc = len(self.names)
        self.metrics.names = self.names
        self.confusion_matrix = ConfusionMatrix(nc=80)
        self.seen = 0
        self.jdict = []
        self.stats = []

