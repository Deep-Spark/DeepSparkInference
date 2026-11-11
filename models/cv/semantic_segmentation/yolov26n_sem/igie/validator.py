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

import tvm
import torch
import numpy as np
from tqdm import tqdm

from ultralytics.models.yolo.semantic import SemanticSegmentationValidator
from ultralytics.utils.metrics import ConfusionMatrix, SemanticMetrics


class IGIE_Validator(SemanticSegmentationValidator):
    def __call__(self, engine, device, imgsz=None):
        self.data = self.args.data
        self.device = torch.device("cpu")
        self.dataloader = self.get_dataloader(self.data.get(self.args.split), self.args.batch)
        # Ultralytics val forces integer imgsz; patch LetterBox back to exact HxW for IGIE engine.
        if imgsz is not None:
            self._set_letterbox_shape(tuple(imgsz))
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
                imgs_np = imgs.numpy() if isinstance(imgs, torch.Tensor) else imgs
                imgs_np = np.resize(imgs_np, (self.args.batch, *imgs_np.shape[1:]))
            else:
                imgs_np = imgs.numpy() if isinstance(imgs, torch.Tensor) else imgs

            engine.set_input(0, tvm.nd.array(imgs_np, device))
            engine.run()

            outputs = engine.get_output(0).asnumpy()
            if pad_batch:
                outputs = outputs[:origin_size]

            preds = self.postprocess(torch.from_numpy(outputs))
            self.update_metrics(preds, batch)

        stats = self.get_stats()
        self.print_results()
        return stats

    def _set_letterbox_shape(self, imgsz):
        """Force dataset LetterBox to a fixed (H, W) matching the exported engine."""
        transforms = getattr(self.dataloader.dataset, "transforms", None)
        if transforms is None:
            return
        items = getattr(transforms, "transforms", [transforms])
        for t in items:
            if hasattr(t, "new_shape"):
                t.new_shape = imgsz

    def preprocess(self, batch):
        """Preprocess without moving images to a CUDA device (TVM consumes host arrays)."""
        batch = self.preprocess_targets(batch)
        batch["img"] = batch["img"].float() / 255.0
        return batch

    def init_metrics(self):
        """Initialize semantic metrics without a PyTorch model instance."""
        self.names = self.data["names"]
        self.nc = len(self.names)
        self.metrics = SemanticMetrics(names=self.names)
        self.seen = 0
        self.dataset = getattr(self.dataloader, "dataset", None)
        self.results_dir = None
        cm_nc = self.metrics.cm_nc
        if cm_nc == 2 and len(self.names) == 1:
            cm_names = {0: "background", 1: next(iter(self.names.values()))}
        else:
            base = list(self.names.values()) + [str(i) for i in range(len(self.names), cm_nc)]
            cm_names = {i: base[i] for i in range(cm_nc)}
        self.confusion_matrix = ConfusionMatrix(names=cm_names, task="semantic")
