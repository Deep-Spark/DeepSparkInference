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

from mmcls.apis import init_model
import argparse
import torch
import onnx

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str)
    parser.add_argument("--checkpoint_file", type=str)
    parser.add_argument("--output_model", type=str)
    args = parser.parse_args()
    return args

device='cuda:0'
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model =  init_model(config_file, checkpoint_file, device='cuda:0') #.switch_to_deploy()
  
    def forward(self, x):
        feat = self.model.backbone(x)
        feat = self.model.neck(feat[0])
        out_head = self.model.head.fc(feat)
        return out_head
    
args = parse_args()
config_file = args.config_file
checkpoint_file = args.checkpoint_file
model = Model().eval()
x = torch.zeros(32, 3, 224, 224).to(device)
with torch.no_grad():
    output = model(x)
  
################ pytorch onnx 模型导出
print ("start transfer model to onnx")
torch.onnx.export(model,
    x,
    args.output_model,
    input_names=["input"],
    output_names=["output"],
    do_constant_folding=True,
    opset_version=12,
)

print ("end transfer model to onnx")

import onnx
import onnxsim
from mmcv import digit_version
  
min_required_version = '0.4.0'
assert digit_version(onnxsim.__version__) >= digit_version(
    min_required_version
), f'Requires to install onnxsim>={min_required_version}'
  
model_opt, check_ok = onnxsim.simplify(args.output_model)
if check_ok:
    onnx.save(model_opt, args.output_model)
    print(f'Successfully simplified ONNX model: {args.output_model}')
else:
    print('Failed to simplify ONNX model.')
