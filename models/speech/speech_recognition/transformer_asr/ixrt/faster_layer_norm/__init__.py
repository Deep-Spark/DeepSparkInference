import torch
from faster_layer_norm import sp_opt

class FasterLayerNorm(torch.nn.Module):
    def __init__(self, weight, bias):
        super(FasterLayerNorm, self).__init__()
        self.weight = weight
        self.bias = bias
    
    def forward(self, inputs, *args, **kwargs):
        hidden_size = self.weight.size(0)
        in_shape = inputs.shape
        inputs = inputs.view(-1,hidden_size)
        output, = sp_opt.test_opt(inputs,self.weight,self.bias)
        output = output.view(*in_shape)
        return output
