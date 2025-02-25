import torch
from faster_stack import sp_opt

# class FasterLogSumExp(torch.nn.Module):
#     def __init__(self, weight, bias):
#         super(FasterLogSumExp, self).__init__()
#         self.weight = weight
#         self.bias = bias
    
#     def forward(self, inputs, *args, **kwargs):
#         hidden_size = self.weight.size(0)
#         in_shape = inputs.shape
#         inputs = inputs.view(-1,hidden_size)
#         output, = sp_opt.test_opt(inputs,self.weight,self.bias)
#         output = output.view(*in_shape)
#         return output

def FasterStack(inputs):
    if len(inputs) == 4:
        a,b,c,d = inputs
        in_shape = a.shape
        res, = sp_opt.test_opt(a.view(-1),b.view(-1),c.view(-1),d.view(-1))
        new_shape = (4,) + in_shape
        res = res.view(*new_shape)
        return res
    if len(inputs) == 2:
        a,b = inputs
        in_shape = a.shape
        res, = sp_opt.test_opt_2(a.view(-1),b.view(-1))
        new_shape = (2,) + in_shape
        res = res.view(*new_shape)
        return res
    return torch.stack(inputs)