import torch
from typing import Optional
from torch.nn.parameter import Parameter
from ixformer.inference.functions.w8a8 import w8a8, dynamic_scaled_int8_quant

from diffusers.models.activations import GELU
def perchannel_quantize_weight_int8(weight: torch.Tensor):
    weight = weight.cpu().to(torch.float32)
    n_bit = 8
    eps = 1e-5
    max_int = 2**(n_bit - 1) - 1
    min_int = -(2**(n_bit - 1)-1)
    max_val = weight.abs().amax(dim=-1, keepdim=True)
    # max_val = max_val.clamp(min=eps)
    qscale = max_val / max_int
    qweight = torch.clamp(torch.round(weight * (1.0 / qscale)), min_int,
                            max_int).to(torch.int8)
    qscale = qscale.squeeze().to(torch.float32)
    return qweight, qscale
class DynamicQuantizeLinear(torch.nn.Module):
    def __init__(self,
                 unquantized: torch.nn.Module,
                 output_dtype: Optional[torch.dtype] = None,
                 ):
        
        super().__init__()
        assert isinstance(unquantized, torch.nn.Linear)
        self.in_features = unquantized.in_features
        self.out_features = unquantized.out_features
        
        self.device = unquantized.weight.device
        self.output_dtype =output_dtype
        
        qweight, qscale = perchannel_quantize_weight_int8(unquantized.weight)
        self.weight = Parameter(qweight.to(self.device), requires_grad=False)
        self.scale = Parameter(qscale.to(self.device), requires_grad=False)

        if unquantized.bias is not None:
            self.bias = unquantized.bias.to(self.device)
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device =self.weight.device
        assert x.device == device
        output_dtype = x.dtype if self.output_dtype is None else self.output_dtype
        inputs = torch.empty(x.shape, dtype=torch.int8, device=device)
        i_scales = torch.empty(x.shape[:-1], dtype=torch.float32, device=device)
        dynamic_scaled_int8_quant(inputs, x.contiguous(), i_scales)

        output = torch.empty(
                (inputs.shape[:-1] + (self.weight.shape[0],)),
                dtype=output_dtype,
                device=device,
            )
        
        out = w8a8(inputs, self.weight, i_scales, self.scale,self.bias, output)
        # if self.bias is not None:
        #     out =out +self.bias
        return out


class DynamicQuantizeGELU(torch.nn.Module):
    def __init__(self,
                 unquantized: torch.nn.Module,
                 output_dtype: Optional[torch.dtype] = None,
                 ):
        
        super().__init__()
        # assert isinstance(unquantized, GELU)

        unquantized_linear = unquantized
        if isinstance(unquantized, GELU):
            unquantized_linear = unquantized.proj
        self.in_features = unquantized_linear.in_features
        self.out_features = unquantized_linear.out_features
        
        self.device = unquantized_linear.weight.device
        self.output_dtype =output_dtype
        
        qweight, qscale = perchannel_quantize_weight_int8(unquantized_linear.weight)
        self.weight = Parameter(qweight.to(self.device), requires_grad=False)
        self.scale = Parameter(qscale.to(self.device), requires_grad=False)

        if unquantized_linear.bias is not None:
            self.bias = unquantized_linear.bias.to(self.device)
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:        
        device =self.weight.device
        assert x.device == device
        output_dtype = x.dtype if self.output_dtype is None else self.output_dtype
        inputs = torch.empty(x.shape, dtype=torch.int8, device=device)
        i_scales = torch.empty(x.shape[:-1], dtype=torch.float32, device=device)
        dynamic_scaled_int8_quant(inputs, x, i_scales)
        # print(f"input {x.shape} weight {self.weight.shape}")
        output = torch.empty(
                (inputs.shape[:-1] + (self.weight.shape[0],)),
                dtype=output_dtype,
                device=device,
            )
        
        out = w8a8(inputs, self.weight, i_scales, self.scale,self.bias, output, act_type=3)

        return out    
def _is_linear(mod, *args):
    # return isinstance(mod, torch.nn.Linear) and args[0] in ["to_qkv", "to_added_qkv", "proj"]
    # if isinstance(mod, torch.nn.Linear):
    #     print(args[0])
    return isinstance(mod, torch.nn.Linear) and "transformer" in args[0]  and ("attn1" in args[0] or "attn" in args[0] or "ff" in args[0] or "proj_mlp" in  args[0] or  "proj_out" in  args[0])
    
def _is_lineargelu_flux(mod, *args):   
    return  "transformer" in args[0]  and ("proj_mlp" in  args[0] or ("net.0" in  args[0] and "net.0.proj" not in  args[0]))
      
def _is_linear_flux(mod, *args):
    # return isinstance(mod, torch.nn.Linear) and args[0] in ["to_qkv", "to_added_qkv", "proj"]
    # if isinstance(mod, torch.nn.Linear):
    #     print(args[0])
    return isinstance(mod, torch.nn.Linear) and "transformer" in args[0] and "net.0" not in  args[0] and ( "attn" in args[0] or "ff" in args[0] or "proj_out" in  args[0] ) 


def apply_quant_linear_i8w8o16(model, cls=DynamicQuantizeLinear, filter_fn = None):
    if filter_fn is None:
        filter_fn = _is_linear
    if type(model).__name__ == "FluxTransformer2DModel" or type(model).__name__ == "xFuserFluxTransformer2DWrapper":        
        filter_fn = _is_linear_flux

    # for name, child in model.named_children():
    #     if filter_fn(child, name):            
    #         setattr(model, name, cls(child))
    #     else:
    #         apply_quant_linear_i8w8o16(child, cls, filter_fn)
    for name, m in model.named_modules():
        if filter_fn(m,name):
            parent_module_name, child_name = name.rsplit('.', 1) if '.' in name else ('', name)
            parent_module = model.get_submodule(parent_module_name)
            # print(parent_module_name,name)
            setattr(parent_module, child_name, cls(m))
    return model
    
def apply_quant_lineargelu_i8w8o16(model, cls=DynamicQuantizeGELU, filter_fn = None):
    if type(model).__name__ == "FluxTransformer2DModel" or type(model).__name__ == "xFuserFluxTransformer2DWrapper":
        filter_fn = _is_lineargelu_flux
    for name, m in model.named_modules():
        if filter_fn(m,name):
            parent_module_name, child_name = name.rsplit('.', 1) if '.' in name else ('', name)
            parent_module = model.get_submodule(parent_module_name)
          
            setattr(parent_module, child_name, cls(m))
    return model