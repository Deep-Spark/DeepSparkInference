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
    batch_tokens = 2
    c = 2
    hidden_size = 320*5000

    inputs = torch.randn([batch_tokens,c, hidden_size]).cuda().half()

    # res1 = torch.log(torch.sum(torch.exp(inputs),dim=-1))
    # res2 = torch.logsumexp(inputs,dim=-1)
    # diff = torch.abs(res1-res2)
    # print(diff.max())

    res_pt = torch.logsumexp(inputs,dim=1)

    res_cu, = sp_opt.test_opt(inputs)

    diff = torch.abs(res_pt - res_cu)
    print(diff.max())

    for i in range(20):
        res_cu, = sp_opt.test_opt(inputs)
    
    batch_tokens = 55
    hidden_size = 320*5000
    inputs = torch.randn([batch_tokens,hidden_size]).cuda().half()
    res_pt = torch.logsumexp(inputs,dim=0)
    res_cu, = sp_opt.test_opt_dim0(inputs)

    diff = torch.abs(res_pt - res_cu)
    print(diff.max())
    for i in range(20):
        res_cu, = sp_opt.test_opt_dim0(inputs)

