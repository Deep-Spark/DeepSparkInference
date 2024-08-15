import torch
from faster_logsumexp import sp_opt

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

def FasterLogSumExp(inputs,dim):
    # print(inputs.shape, dim)
    if dim == 1 and len(inputs.shape)>2 and inputs.size(1)==2:
        in_shape = inputs.shape
        inputs = inputs.view(in_shape[0],in_shape[1],-1)
        res, = sp_opt.test_opt(inputs)
        new_shape = (in_shape[0],) + in_shape[2:]
        res = res.view(*new_shape)
        return res
    # dim==0 现在的实现会有bug?
    # if dim == 0 and len(inputs.shape)>=2:
    #     in_shape = inputs.shape
    #     inputs = inputs.view(in_shape[0],-1)
    #     res, = sp_opt.test_opt_dim0(inputs)
    #     new_shape = in_shape[1:]
    #     res = res.view(*new_shape)
    #     return res
    # print(f"not support shape: {inputs.shape} dim: {dim}")
    res = torch.logsumexp(inputs, dim=dim)
    return res
        
