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

from mmdet.apis import init_detector
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import torch
import onnx
import argparse

class Model(torch.nn.Module):
    def __init__(self,config_file,checkpoint_file):
        super().__init__()
        self.model =  init_detector(config_file, checkpoint_file, device='cuda:0')

    def forward(self, x):
        feat = self.model.backbone(x)
        out_neck =self.model.neck(feat)
        out_head =self.model.mask_head(out_neck)
        return out_head

def parse_args():
    parser = argparse.ArgumentParser()
    # engine args
    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--input_size", type=tuple, default=(800,800))
    args = parser.parse_args()
    return args
    
    
def main():
    args= parse_args()
    model = Model(args.cfg,args.checkpoint)
    model.eval()
    device='cuda:0'
    
    input = torch.zeros(args.batch_size, 3, args.input_size[0], args.input_size[1]).to(device)
    
    
    output_onnx_name = f'r50_solo_bs{args.batch_size}_{args.input_size[0]}x{args.input_size[1]}.onnx'

    # ################ pytorch onnx 模型导出
    print ("start transfer model to onnx")
    torch.onnx.export(model,
        input,
        output_onnx_name,
        input_names=["input"],
        output_names=["output"],
        do_constant_folding=True,
        opset_version=11,
        keep_initializers_as_inputs=True,
        # dynamic_axes={'input':{0:'batch'}, 'output':{0:'batch'}}
        # dynamic_axes={'input':{0:'batch', 2:'h', 3:'w'}, 'output':{0:'batch', 2:'h2', 3:'w2'}}
    )
    print ("end transfer model to onnx")
    
    output_file =output_onnx_name
    import onnx
    import onnxsim
    from mmcv import digit_version
    
    min_required_version = '0.4.0'
    assert digit_version(onnxsim.__version__) >= digit_version(
        min_required_version
    ), f'Requires to install onnxsim>={min_required_version}'
    
    model_opt, check_ok = onnxsim.simplify(output_file)
    if check_ok:
        onnx.save(model_opt, output_file)
        print(f'Successfully simplified ONNX model: {output_file}')
    else:
        print('Failed to simplify ONNX model.')
        
        
if __name__ == "__main__":
    main()       
    
    
    
        

