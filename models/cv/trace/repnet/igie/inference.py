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
import argparse
import tvm
import torch
import torchvision
import numpy as np
from tvm import relay
from tqdm import tqdm
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--engine", 
                        type=str, 
                        required=True, 
                        help="igie engine path.")
    
    parser.add_argument("--batchsize",
                        type=int,
                        required=True, 
                        help="inference batch size.")
    
    parser.add_argument("--datasets", 
                        type=str, 
                        required=True, 
                        help="datasets path.")

    parser.add_argument("--input_name", 
                        type=str, 
                        required=True, 
                        help="input name of the model.")
    
    parser.add_argument("--warmup", 
                        type=int, 
                        default=3, 
                        help="number of warmup before test.")           
    
    parser.add_argument("--num_workers",
                        type=int,
                        default=16,
                        help="number of workers used in pytorch dataloader.")
    
    parser.add_argument("--acc_target",
                        type=float,
                        default=None,
                        help="Model inference Accuracy target.")
    
    parser.add_argument("--fps_target",
                        type=float,
                        default=None,
                        help="Model inference FPS target.")

    parser.add_argument("--perf_only",
                        type=bool,
                        default=False,
                        help="Run performance test only")
    
    args = parser.parse_args()

    return args

class VehicleDatasets(torch.utils.data.Dataset):
    def __init__(self, image_list, image_size=(224, 224)):
        super().__init__()
        self.image_dir_path = image_list
        self.images = []
        self.length = 0

        for img_path in self.image_dir_path:
            self.images.append(img_path)
            self.length += 1
        
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
        ])

    def preprocess(self, image_path):
        img = Image.open(image_path)
        
        if img.mode == 'L' or img.mode == 'I':
            img = img.conver("RGB")
        
        img = self.transforms(img)

        return img
    
    def __getitem__(self, index):
        image = self.preprocess(self.images[index])
        image_name = self.images[index].split('/')[-1].strip()
        return image, image_name

    def __len__(self):
        return self.length
    
def get_dataset(img_dir, pair_set_txt='pair_set_vehicle.txt'):
    pairs, imgs_path = [], []
    try:
        with open(pair_set_txt, 'r') as fh:
            for line in fh.readlines():
                pair = line.strip().split()

                imgs_path.append(os.path.join(img_dir, 'image', pair[0] + '.jpg'))
                imgs_path.append(os.path.join(img_dir, 'image', pair[1] + '.jpg'))

                pairs.append(pair)
        
    except FileNotFoundError:
        assert os.path.isfile(pair_set_txt), "File does not exist." 
    
    list(set(imgs_path)).sort()

    dataset = VehicleDatasets(imgs_path)

    return dataset, pairs

def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

def cal_accuracy(y_score, y_true):
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    best_acc = 0
    best_th = 0
    
    for i in range(len(y_score)):
        th = y_score[i]
        y_test = (y_score >= th)
        acc = np.mean((y_test == y_true).astype(int))

        if acc > best_acc:
            best_acc = acc
            best_th = th

    return best_acc, best_th

def main():
    args = parse_args()

    batch_size = args.batchsize

    # create iluvatar target & device
    target = tvm.target.iluvatar(model="MR", options="-libs=cudnn,cublas,ixinfer")    
    device = tvm.device(target.kind.name, 0)

    # load engine
    lib = tvm.runtime.load_module(args.engine)

    # create runtime from engine
    module = tvm.contrib.graph_executor.GraphModule(lib["default"](device))

    # just run perf test
    if args.perf_only:
        ftimer = module.module.time_evaluator("run", device, number=100, repeat=1)        
        prof_res = np.array(ftimer().results) * 1000 
        fps = batch_size * 1000 / np.mean(prof_res)
        print(f"\n* Mean inference time: {np.mean(prof_res):.3f} ms, Mean fps: {fps:.3f}")
    else:
        # warm up
        for _ in range(args.warmup):
            module.run()

        # get dataloader
        dataset, pairs = get_dataset(args.datasets)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size, num_workers=args.num_workers, drop_last=True)

        features_map = {}
        
        for inputs, path in tqdm(dataloader):
            # Pad the last batch
            pad_batch = len(inputs) != batch_size
            if pad_batch:
                origin_size = len(inputs)
                inputs = np.resize(inputs, (batch_size, *inputs.shape[1:]))

            module.set_input('input', tvm.nd.array(inputs, device))
            
            module.run()

            features = torch.from_numpy(module.get_output(0).asnumpy())

            if pad_batch:
                features = features[:origin_size]

            for index, output in enumerate(features):
                features_map[path[index]] = output


        sims, labels = [], []
        for pair in pairs:
            sim = cosin_metric(features_map[pair[0] + '.jpg'], features_map[pair[1] + '.jpg'])
            label = int(pair[2])
            sims.append(sim)
            labels.append(label)

        acc, th = cal_accuracy(sims, labels)
        print('=> best accuracy: %.3f %%, at threshold: %.3f' % (acc * 100.0, th))

if __name__ == "__main__":
    main()