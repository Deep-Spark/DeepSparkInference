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
import onnx
import psutil
import argparse
import numpy as np
from pathlib import Path

import torch

from onnxruntime.quantization import (CalibrationDataReader, QuantFormat,
                                      quantize_static, QuantType,
                                      CalibrationMethod)

from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect import DetectionValidator

class CalibrationDataLoader(CalibrationDataReader):
    def __init__(self, input_name, dataloader, cnt_limit=100):
        self.cnt = 0
        self.input_name = input_name
        self.cnt_limit = cnt_limit
        self.dataloader = dataloader
        self.iter = iter(dataloader)

    # avoid oom
    @staticmethod
    def _exceed_memory_upper_bound(upper_bound=80):
        info = psutil.virtual_memory()
        total_percent = info.percent
        if total_percent >= upper_bound:
            return True
        return False

    def get_next(self):
        if self._exceed_memory_upper_bound() or self.cnt >= self.cnt_limit:
            return None
        self.cnt += 1
        print(f"onnx calibration data count: {self.cnt}")
        input_info = next(self.iter)

        ort_input = {self.input_name[0]: input_info.numpy()}
        
        return ort_input

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", 
                        type=str, 
                        required=True, 
                        help="original model path.")
    
    parser.add_argument("--out_path", 
                        type=str, 
                        required=True, 
                        help="igie export engine path.")

    parser.add_argument("--datasets", 
                        type=str, 
                        required=True, 
                        help="calibration datasets path.")
    
    parser.add_argument("--batch",
                        type=int,
                        default=32,
                        help="batchsize of the model.")
            
    args = parser.parse_args()

    return args

class PreProcessDatasets(DetectionValidator):
    def __call__(self, data):
        self.data = data
        self.stride = 32
        self.dataloader = self.get_dataloader(self.data.get(self.args.split), self.args.batch)

        datasets = []
        length = 0

        for batch in self.dataloader:
            data = self.preprocess(batch)['img']
            datasets.append(data[0])
            length += data.shape[0]

            if length >= 200:
                break

        return datasets

class CalibrationDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, index):
        return self.datasets[index]


def main():
    args = parse_args()

    model = onnx.load(args.model_path)
    input_names = [input.name for input in model.graph.input]

    overrides = {'mode': 'val'}
    cfg_args = get_cfg(cfg=DEFAULT_CFG, overrides=overrides)

    cfg_args.batch = 1
    cfg_args.save_json = True

    data = {
        'path': Path(args.datasets),
        'val': os.path.join(args.datasets, 'val2017.txt')
    }
    
    validator = PreProcessDatasets(args=cfg_args, save_dir=Path('.'))

    datasets = CalibrationDataset(validator(data))
    
    data_loader = torch.utils.data.DataLoader(dataset=datasets, batch_size=args.batch)

    cnt_limit = int(20 / args.batch) + 1
    
    calibration = CalibrationDataLoader(input_names, data_loader, cnt_limit=cnt_limit)
    
    quantize_static(args.model_path,
                args.out_path,
                calibration_data_reader=calibration,
                quant_format=QuantFormat.QOperator,
                op_types_to_quantize=['Conv'],
                per_channel=False,
                activation_type=QuantType.QInt8,
                weight_type=QuantType.QInt8,
                use_external_data_format=False,
                nodes_to_exclude=['/model.22/Add_10', '/model.22/Add_11', '/model.22/Add_9', '/model.22/Concat_24', '/model.22/Concat_25', '/model.22/Mul_4', '/model.22/Mul_5', '/model.22/dfl/Softmax'],
                calibrate_method=CalibrationMethod.Percentile,
                extra_options = {
                    'ActivationSymmetric': True, 
                    'WeightSymmetric': True
                }
    )
    
if __name__ == "__main__":
    main()