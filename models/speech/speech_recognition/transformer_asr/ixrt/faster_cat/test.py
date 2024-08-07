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
import sp_opt

if __name__ == "__main__":
    m1 = 320
    m2 = 321
    hidden_size = 5000

    a = torch.randn([m1,hidden_size]).cuda().half()
    b = torch.randn([m2,hidden_size]).cuda().half()
   

    res_pt = torch.cat([a,b],dim=0)
    
    res_cu, = sp_opt.test_opt_2(a,b)
    

    diff = torch.abs(res_pt-res_cu)
    print(diff)
    print(diff.max())

    for i in range(20):
        res_cu, = sp_opt.test_opt_2(a,b)