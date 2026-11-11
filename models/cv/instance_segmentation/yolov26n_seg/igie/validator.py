# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
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
import json

import tvm
import torch
import numpy as np
from tqdm import tqdm

from ultralytics.models.yolo.segment import SegmentationValidator
from ultralytics.utils import ops
from ultralytics.utils.metrics import ConfusionMatrix, SegmentMetrics
from ultralytics.data.converter import coco80_to_coco91_class


class IGIE_Validator(SegmentationValidator):
    def __call__(self, engine, device):
        self.data = self.args.data
        self.device = torch.device("cpu")
        self.dataloader = self.get_dataloader(self.data.get(self.args.split), self.args.batch)
        self.init_metrics()

        # warm up
        for _ in range(3):
            engine.run()

        for batch in tqdm(self.dataloader):
            batch = self.preprocess(batch)

            imgs = batch["img"]
            pad_batch = len(imgs) != self.args.batch
            if pad_batch:
                origin_size = len(imgs)
                imgs = np.resize(imgs.numpy() if isinstance(imgs, torch.Tensor) else imgs,
                                 (self.args.batch, *imgs.shape[1:]))
            else:
                imgs = imgs.numpy() if isinstance(imgs, torch.Tensor) else imgs

            engine.set_input(0, tvm.nd.array(imgs, device))
            engine.run()

            out0 = engine.get_output(0).asnumpy()
            out1 = engine.get_output(1).asnumpy()

            if pad_batch:
                out0 = out0[:origin_size]
                out1 = out1[:origin_size]

            preds = self.postprocess([torch.from_numpy(out0), torch.from_numpy(out1)])
            self.update_metrics(preds, batch)

        stats = self.get_stats()
        self.print_results()

        if self.args.save_json and self.jdict:
            with open(str(self.save_dir / "predictions.json"), "w") as f:
                print(f"Saving {f.name} ...")
                json.dump(self.jdict, f)
            stats = self.eval_json(stats)

        return stats

    def preprocess(self, batch):
        """Preprocess without moving images to a CUDA device (TVM consumes host arrays)."""
        batch = self.preprocess_targets(batch)
        batch["img"] = batch["img"].float() / 255.0
        return batch

    def init_metrics(self):
        """Initialize evaluation metrics without a PyTorch model instance."""
        val = self.data.get(self.args.split, "")
        self.is_coco = isinstance(val, str) and "coco" in val and val.endswith(f"{os.sep}val2017.txt")
        self.is_lvis = False
        self.class_map = coco80_to_coco91_class() if self.is_coco else list(range(1000))
        self.args.save_json |= self.is_coco and not self.training
        self.names = self.data["names"]
        self.nc = len(self.names)
        # YOLO26-seg ONNX export keeps classic (B, 4+nc+nm, anchors) outputs + proto.
        self.end2end = False
        self.seen = 0
        self.jdict = []
        self.is_custom_json = False
        self.gdict = None
        self.build_gdict = False
        self.eval_ids = None
        self.pred_counts = []
        self.metrics = SegmentMetrics()
        self.metrics.names = self.names
        self.metrics.clear_stats()
        if hasattr(self.metrics, "clear_image_metrics"):
            self.metrics.clear_image_metrics()
        self.confusion_matrix = ConfusionMatrix(names=self.names)
        self.process = ops.process_mask
