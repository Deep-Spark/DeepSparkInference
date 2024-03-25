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
import torch
import torchvision.models as models
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin_model", type=str)
    parser.add_argument("--output_model", type=str)
    args = parser.parse_args()
    return args

args = parse_args()
model = models.alexnet()
model.load_state_dict(torch.load(args.origin_model))
model.cuda()
model.eval()
input = torch.randn(1, 3, 224, 224, device='cuda')
export_onnx_file = args.output_model

torch.onnx.export(model,        
                  input,            
                  export_onnx_file,       
                  export_params=True,  
                  opset_version=11,    
                  do_constant_folding=True,  
                  input_names = ['input'],   
                  output_names = ['output'],) 
print(" ") 
print('Model has been converted to ONNX') 
print("exit")
exit()