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

import time

import requests
import torch

# from transformers import CLIPModel
from ixformer.inference.models.clip import CLIPModel
from PIL import Image
from torch.cuda import profiler
from transformers import CLIPProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = (
    CLIPModel.from_pretrained("data/clip-vit-base-patch32")
    .to(device)
    .half()
)
model = model.eval()
processor = CLIPProcessor.from_pretrained("data/clip-vit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

metricResult = {"metricResult": {}}
batch_size_list = [32, 64, 128, 256, 512, 1024, 2048]
with torch.no_grad():
    e2e_start_time = time.time()
    for batch_size in batch_size_list:
        images = [image for item in range(batch_size)]
        inputs = processor(
            text=["a photo of a cat", "a photo of a dog"],
            images=images,
            return_tensors="pt",
            padding=True,
        )
        inputs["input_ids"] = inputs["input_ids"].to(device)
        inputs["attention_mask"] = inputs["attention_mask"].to(device)
        inputs["pixel_values"] = inputs["pixel_values"].to(device).half()
        # warmup
        for i in range(2):
            outputs = model(**inputs)
        torch.cuda.synchronize()
        profiler.start()
        start_time = time.perf_counter()
        outputs = model(**inputs)
        profiler.stop()
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        logits_per_image = (
            outputs.logits_per_image
        )  # this is the image-text similarity score
        probs = logits_per_image.softmax(
            dim=1
        )  # we can take the softmax to get the label probabilities
        print(probs[:5])
        print(probs[-5:-1])
        metricResult["metricResult"][f"QPS-batch_size-{batch_size}"] = round(batch_size / (end_time - start_time), 3)
        print("QPS: ", batch_size / (end_time - start_time))
    e2e_time = time.time() - e2e_start_time
    metricResult["metricResult"]["E2E time"] = round(e2e_time, 3)
    print(metricResult)