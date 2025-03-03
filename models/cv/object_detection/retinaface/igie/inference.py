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
import tvm
import torch
import argparse
import numpy as np
from tvm import relay
from tqdm import tqdm
from torch.utils.data import Dataset
from utils.post_process import post_process

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

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = 1

    # Compute padding
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1] 
    dw /= 2 
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im,
                            top,
                            bottom,
                            left,
                            right,
                            cv2.BORDER_CONSTANT,
                            value=color)
    return im, ratio, (dw, dh)


class FaceDataset(Dataset):
    def __init__(self, img_path,image_size=320, layout="NCHW"):
  
        self.imgs_path = []
        self.imgs_path_ori=[]
        self.image_size=image_size
        self.layout = layout
        self.img_dir=os.path.dirname(img_path)
        with open(img_path, 'r') as fr:
            self.imgs_path = fr.read().split()
        self.imgs_path_ori=self.imgs_path

    def __len__(self):
        return len(self.imgs_path)  
   
    def __getitem__(self, idx):
        img, (h0, w0), (h, w) = self._load_image(idx)
        img, ratio, pad = letterbox(img,
                                    self.image_size,
                                    color=(114,114,114))
        shapes = (h0, w0), ((h / h0, w / w0), pad),(h, w)
        img = img.astype(np.float32)
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)

        return img, self.imgs_path[idx], shapes, self.imgs_path_ori[idx]

    
    @staticmethod
    def collate_fn(batch):
        im, path, shapes, path_ori = zip(*batch)
        return np.concatenate([i[None] for i in im], axis=0), path, shapes, path_ori

    def _load_image(self, i):
        im = cv2.imread(self.img_dir+'/images'+self.imgs_path[i], cv2.IMREAD_COLOR)
        h0, w0 = im.shape[:2] 
        r = self.image_size / max(h0, w0)  
        if r != 1:  
            im = cv2.resize(im, (int(w0 * r), int(h0 * r)),
                            interpolation=cv2.INTER_LINEAR)
        return im.astype("float32"), (h0, w0), im.shape[:2]

def get_dataloader(args):
    image_size = 320
    batchsize = args.batchsize
    data_path = os.path.join(args.datasets, 'val/wider_val.txt')
    datasets =FaceDataset(data_path, image_size)
    dataLoader = torch.utils.data.DataLoader(datasets, batchsize, drop_last=False, collate_fn=datasets.collate_fn)

    return dataLoader

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
        # warmup
        for _ in range(args.warmup):
            module.run()
        
        dataloader = get_dataloader(args)

        for batch in tqdm(dataloader):
            image = batch[0]
            shapes = batch[2]
            img_names = batch[3]

            pad_batch = len(image) != batch_size
            if pad_batch:
                origin_size = len(image)
                image = np.resize(image, (batch_size, *image.shape[1:]))

            module.set_input("input", tvm.nd.array(image, device))
        
            module.run()

            loc_bs, conf_bs, landms_bs = module.get_output(0).asnumpy(), module.get_output(1).asnumpy(), module.get_output(2).asnumpy()

            if pad_batch:
                loc_bs = loc_bs[:origin_size]
                conf_bs = conf_bs[:origin_size]
                landms_bs = landms_bs[:origin_size]

            ## batch accuracy
            post_process(shapes, img_names, loc_bs, conf_bs, landms_bs, save_folder='./widerface_evaluate/widerface_txt/')
    
if __name__ == "__main__":
    main()
