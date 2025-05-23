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
import onnx
import psutil
import argparse
import numpy as np
from inference import get_dataloader
from onnxruntime.quantization import (CalibrationDataReader, QuantFormat,
                                      quantize_static, QuantType,
                                      CalibrationMethod)

class CalibrationDataLoader(CalibrationDataReader):
    def __init__(self, input_name, dataloader, cnt_limit=100):
        self.cnt = 0
        self.input_name = input_name
        self.cnt_limit = cnt_limit
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

        ort_input = {k: np.array(v) for k, v in zip(self.input_name, input_info)}
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

    parser.add_argument("--num_workers",
                    type=int,
                    default=16,
                    help="number of workers used in pytorch dataloader.")
            
    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    model = onnx.load(args.model_path)
    input_names = [input.name for input in model.graph.input]

    data_path = os.path.join(args.datasets, "images", "val2017")
    label_path = os.path.join(args.datasets, "annotations", "instances_val2017.json")
    
    dataloader = get_dataloader(data_path, label_path, batch_size=1, num_workers=args.num_workers)
    calibration = CalibrationDataLoader(input_names, dataloader, cnt_limit=20)
    
    quantize_static(args.model_path,
                args.out_path,
                calibration_data_reader=calibration,
                quant_format=QuantFormat.QOperator,
                per_channel=False,
                activation_type=QuantType.QInt8,
                weight_type=QuantType.QInt8,
                use_external_data_format=False,
                nodes_to_exclude= [
                    '/Concat_10', '/Concat_11', '/models.139/Add', '/models.139/Add_1', '/models.139/Add_2', '/models.139/Add_3', '/models.139/Add_4', 
                    '/models.139/Add_5', '/models.139/Add_6', '/models.139/Add_7', '/models.139/Concat', '/models.139/Concat_1', '/models.139/Concat_14', 
                    '/models.139/Concat_2', '/models.139/Concat_4', '/models.139/Concat_7', '/models.139/Concat_8', '/models.139/Mul_10', '/models.139/Mul_11', 
                    '/models.139/Mul_12', '/models.139/Mul_3', '/models.139/Mul_4', '/models.139/Mul_5', '/models.139/Mul_6', '/models.139/Mul_7', '/models.139/Mul_8', 
                    '/models.139/Mul_9', '/models.150/Add', '/models.150/Add_1', '/models.150/Add_2', '/models.150/Add_3', '/models.150/Add_4', '/models.150/Add_5', 
                    '/models.150/Add_6', '/models.150/Add_7', '/models.150/Concat', '/models.150/Concat_1', '/models.150/Concat_14', '/models.150/Concat_2', 
                    '/models.150/Concat_4', '/models.150/Concat_7', '/models.150/Concat_8', '/models.150/Mul_10', '/models.150/Mul_11', '/models.150/Mul_12', 
                    '/models.150/Mul_3', '/models.150/Mul_4', '/models.150/Mul_5', '/models.150/Mul_6', '/models.150/Mul_7', '/models.150/Mul_8', '/models.150/Mul_9', 
                    '/models.161/Add', '/models.161/Add_1', '/models.161/Add_2', '/models.161/Add_3', '/models.161/Add_4', '/models.161/Add_5', '/models.161/Add_6', 
                    '/models.161/Add_7', '/models.161/Concat', '/models.161/Concat_1', '/models.161/Concat_14', '/models.161/Concat_2', '/models.161/Concat_4', 
                    '/models.161/Concat_7', '/models.161/Concat_8', '/models.161/Mul_10', '/models.161/Mul_11', '/models.161/Mul_12', '/models.161/Mul_3', 
                    '/models.161/Mul_4', '/models.161/Mul_5', '/models.161/Mul_6', '/models.161/Mul_7', '/models.161/Mul_8', '/models.161/Mul_9',
                    '/models.149/conv102/Conv', '/models.160/conv110/Conv', '/models.138/conv94/Conv'
                ],
                calibrate_method=CalibrationMethod.Percentile,
                extra_options = {
                    'ActivationSymmetric': True, 
                    'WeightSymmetric': True
                }
    )
    
if __name__ == "__main__":
    main()