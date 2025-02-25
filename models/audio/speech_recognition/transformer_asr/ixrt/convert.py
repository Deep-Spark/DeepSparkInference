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
from faster_layer_norm import FasterLayerNorm

def replace_layer_norm(model):
    module_output = model

    if isinstance(model, torch.nn.modules.normalization.LayerNorm):
        return FasterLayerNorm(model.weight, model.bias)

    for name, child in model.named_children():
        module_output.add_module(
            name, replace_layer_norm(child)
        )
    return module_output


def convert_decoder_model(model):
    model = replace_layer_norm(model)
    # for layer in model.layers:
    #     norm = layer.norm1.norm
    #     print(type(norm))
    #     exit()
    #     new_norm = FasterLayerNorm(norm.weight, norm.bias)
    #     layer.norm1.norm = new_norm

    #     norm = layer.norm2.norm
    #     new_norm = FasterLayerNorm(norm.weight, norm.bias)
    #     layer.norm2.norm = new_norm

    #     norm = layer.norm3.norm
    #     new_norm = FasterLayerNorm(norm.weight, norm.bias)
    #     layer.norm3.norm = new_norm
    return model

# def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
#     if type(module) in layers:
#         return {name: module}
#     res = {}
#     for name1, child in module.named_children():
#         res.update(find_layers(
#             child, layers=layers, name=name + '.' + name1 if name != '' else name1
#         ))
#     return res

def find_node(module):
    if type(module) in [torch.nn.LayerNorm]:
        print(module)
        return
    res = {}
    for name1, child in module.named_children():
        find_node(child)
    return


def patch_get_lookahead_mask(padded_input):
    """Creates a binary mask for each sequence which maskes future frames.

    Arguments
    ---------
    padded_input: torch.Tensor
        Padded input tensor.

    Example
    -------
    >>> a = torch.LongTensor([[1,1,0], [2,3,0], [4,5,0]])
    >>> get_lookahead_mask(a)
    tensor([[0., -inf, -inf],
            [0., 0., -inf],
            [0., 0., 0.]])
    """
    seq_len = padded_input.shape[1]
    mask = (
            torch.triu(torch.ones((seq_len, seq_len), device=padded_input.device))
            == 1
    ).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask.detach().to(padded_input.device).to(torch.float16)