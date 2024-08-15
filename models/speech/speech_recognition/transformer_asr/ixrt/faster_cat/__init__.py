import torch
from faster_cat import sp_opt

def FastCat(inputs,dim=0):
    if len(inputs) == 2 and dim==0:
        a,b = inputs
        in_shape = a.shape
        if len(in_shape)>1:
            res, = sp_opt.test_opt_2(a.view(a.shape[0],-1),b.view(b.shape[0],-1))
            new_shape = (a.shape[0]+b.shape[0],) + in_shape[1:]
            res = res.view(*new_shape)
            return res
    return torch.cat(inputs,dim=dim)