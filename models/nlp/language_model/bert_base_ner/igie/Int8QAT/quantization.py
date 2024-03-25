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

import torch.nn.functional as F
from pytorch_quantization.nn.modules.tensor_quantizer import (
    TensorQuantizer,
    disable_quant,
    enable_quant,
    ptq_mode,
    qat_mode,
)
from pytorch_quantization.tensor_quant import (
    QUANT_DESC_8BIT_PER_TENSOR,
    QuantDescriptor,
)
from torch.nn import Linear

act_quant_config = QuantDescriptor(
    num_bits=8, narrow_range=True, learn_amax=False, amax=16.0
)
out_quant_config = QuantDescriptor(
    num_bits=8, narrow_range=True, learn_amax=False, amax=16.0
)
relu_quant_config = QuantDescriptor(
    num_bits=8, narrow_range=True, learn_amax=False, amax=16.0, unsigned=True
)
weight_quant_config = QuantDescriptor(
    num_bits=8, narrow_range=True, learn_amax=False, amax=1.0
)


class QuantLinear(Linear):
    def __init__(self, in_features, out_features, pre_activation=None, *args, **kwargs):
        super(QuantLinear, self).__init__(in_features, out_features, *args, **kwargs)
        if pre_activation == "relu":
            input_quant_config = relu_quant_config
        else:
            input_quant_config = act_quant_config

        self.input_quant = None
        if pre_activation != "encoder_out":
            self.input_quant = TensorQuantizer(input_quant_config)
        self.output_quant = None
        # if pre_activation is None:
        self.output_quant = TensorQuantizer(out_quant_config)
        self.weight_quant = TensorQuantizer(weight_quant_config)

    def forward(self, input):
        qinput = input
        if self.input_quant is not None:
            qinput = self.input_quant(input)
        qweight = self.weight_quant(self.weight)
        output = F.linear(qinput, qweight)
        if self.output_quant is not None:
            output = self.output_quant(output)
        if self.bias is not None:
            output = output + self.bias

        return output
